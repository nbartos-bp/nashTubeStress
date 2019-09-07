#!/usr/bin/env python
# Copyright (C) 2018 William R. Logie

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
nashTubeStress.py
 -- steady-state temperature distribution (Gauss-Seidel iteration)
 -- biharmonic thermoelastic stress
 -- tested 09/07/2019 with Python 2.7.15+ and pip packages 

See also:
 -- Solar Energy 160 (2018) 368-379
 -- https://doi.org/10.1016/j.solener.2017.12.003
"""

import sys, time, os
from math import exp, log, sqrt, pi, ceil, floor
import numpy as np # version 1.15.4
from numpy import ma
import weave # version 0.17.0
from weave import converters
import scipy.optimize as opt # version 1.1.0

# Plotting:
import matplotlib as mpl # version 2.2.3
import matplotlib.pyplot as plt
params = {'text.latex.preamble': [r'\usepackage{mathptmx,txfonts}']}
plt.rcParams.update(params)
mpl.rc('figure.subplot', bottom=0.13, top=0.95)
mpl.rc('figure.subplot', left=0.15, right=0.95)
mpl.rc('xtick', labelsize='medium')
mpl.rc('ytick', labelsize='medium')
mpl.rc('axes', labelsize='large')
mpl.rc('axes', titlesize='large')
mpl.rc('legend', fontsize='medium')
mpl.rc('lines', markersize=4)
mpl.rc('lines', linewidth=0.5)
from matplotlib import colors, ticker, cm
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.projections import PolarAxes
import matplotlib.transforms as mtransforms
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist.grid_finder import \
    (FixedLocator, MaxNLocator, DictFormatter)
# if you're matplotlib is older than version 2:
#import colormaps as cmaps # magma, inferno, plasma, viridis

""" ________________________ PLOTTING FUNCTIONS _______________________ """

def plotStress(theta, r, sigma, sigmaMin, sigmaMax, filename):
    fig = plt.figure(figsize=(2.5, 3))    
    fig.subplots_adjust(left=-1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    cmap = cm.get_cmap('magma')
    #cmap = cmaps.magma
    levels = ticker.MaxNLocator(nbins=10).tick_values(
        sigmaMin*1e-6, sigmaMax*1e-6
    )
    cf = ax.contourf(theta, r, sigma*1e-6, levels=levels, cmap=cmap)
    ax.set_rmin(0)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label('$\sigma$ [MPa]')
    ax.patch.set_visible(False)
    ax.spines['polar'].set_visible(False)
    gridlines = ax.get_xgridlines()
    ticklabels = ax.get_xticklabels()
    for i in range(5, len(gridlines)):
        gridlines[i].set_visible(False)
        ticklabels[i].set_visible(False)
    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x', linewidth=0.2)
    plt.setp(ax.get_yticklabels(), visible=False)
    #fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

""" ____________________________ FUNCTIONS ____________________________ """

def fourierTheta(theta, a0, *c):
    """ Timoshenko & Goodier equation """
    ret = a0 + np.zeros(len(theta))
    for i, n in zip(range(0,len(c),2), range(1,(len(c)/2)+1)):
        ret += (c[i] * np.cos(n * theta)) + (c[i+1] * np.sin(n * theta))
    return ret

def HTC(debug, thermo, a, b, k, mode, arg):
    """
    Inputs:
        debug : (default:False)
        thermo : liquidSodium, nitrateSalt, chlorideSalt
        a : tube inner diameter (m)
        b : tube outer diameter (m)
        k : tube thermal conductivity (W/(m.K))
        mode : 'velocity','mdot','heatCapRate' (m/s,kg/s,???)
        arg : either velocity, mass-flow or heat capacity rate
    Output:
        h : heat transfer coefficient (W/(m^2.K))
    """
    d_i = a*2 # inner pipe diameter [m]
    t = b - a # tube wall thickness [m]
    A_d_i = pi * pow(d_i/2., 2) # cross sectional area of pipe flow
    if mode=='velocity':
        U = arg # m/s
        mdot = U * (A_d_i * thermo.rho)
        hcr = mdot * thermo.Cp
    elif mode=='mdot':
        mdot = arg # kg/s
        U = mdot / (A_d_i * thermo.rho)
        hcr = mdot * thermo.Cp
    elif mode=='heatCapRate':
        hcr = arg # 
        mdot = hcr / thermo.Cp
        U = mdot / (A_d_i * thermo.rho)
    else: sys.exit('Incorrect mode in HTC(mode, thermo, a, arg)!')
    Re = U * d_i / thermo.nu
    f = pow(0.790 * np.log(Re) - 1.64, -2)
    DP_f = -f * (0.5 * thermo.rho * pow(U, 2)) \
              / d_i # kg/m/s^2 for a metre of pipe!
    if isinstance(thermo, liquidSodium):
        # Skupinshi, Tortel and Vautrey (Holman p318):
        Nu = 4.82 + 0.0185 * pow(Re * thermo.Pr, 0.827)
    else:
        # Dittus-Boelter (Holman p286):
        Nu = 0.023 * pow(Re, 0.8) * pow(thermo.Pr, 0.4)
    h = Nu * thermo.kappa / d_i
    Bi = (t * h) / k
    if debug==True:
        valprint('U', U, 'm/s')
        valprint('mdot', mdot, 'kg/s')
        valprint('deltaP', DP_f, 'Pa/m')
        valprint('HCR', hcr, 'J/K/s')
        valprint('h_int', h, 'W/m^2/K')
        valprint('Bi', Bi)
    return h

def headerprint(string, mychar='='):
    """ Prints a centered string to divide output sections. """
    mywidth = 64
    numspaces = mywidth - len(string)
    before = int(ceil(float(mywidth-len(string))/2))
    after  = int(floor(float(mywidth-len(string))/2))
    print("\n"+before*mychar+string+after*mychar+"\n")

def valprint(string, value, unit='-'):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .4f} ({2})".format(string, value, unit))

""" ________________________ CLASS DEFINITIONS ________________________ """

class liquidSodium:
    """
    Usage: thermo = liquidSodium()
           thermo.update(T) # T in [K]
    """
    def __init__ (self, debug):
        self.debug = debug
        if debug==True:
            headerprint('Liquid Sodium', ' ')

    def update (self, T):
        self.T = T
        T_c = 2503.7 # K            
        rho_c = 219. # kg/m^3
        self.rho = rho_c + 275.32*(1 - self.T/T_c) + \
                   511.58*sqrt(1 - self.T/T_c) # kg/m^3
        self.Cp = (1.6582 - 8.4790e-4*self.T + \
                   4.4541e-7*pow(self.T, 2) - \
                   2992.6*pow(self.T, -2) ) *1e3 # m^2/s^2/K
        self.mu = exp(-6.4406 - 0.3958*log(self.T) + \
                      556.835/self.T)# kg/m/s
        self.kappa = 124.67 - 0.11381*self.T + \
                     5.5226e-5*pow(self.T, 2) - \
                     1.1842e-8*pow(self.T, 3) # kg*m/s^3/K
        self.nu = self.mu / self.rho
        self.alpha = self.kappa / (self.rho * self.Cp)
        self.Pr = self.nu / self.alpha
        if self.debug==True:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

class nitrateSalt:
    """
    Usage: thermo = nitrateSalt()
           thermo.update(T) # T in [K]
    """
    def __init__ (self, debug):
        self.debug = debug
        if debug==True:
            headerprint('Nitrate Salt', ' ')

    def update (self, T):
        self.T = min(T, 873.15) # K
        self.rho = 2263.7234 - 0.636*self.T # kg/m^3
        self.Cp = 1396.0182 + 0.172*self.T # m^2/s^2/K
        self.mu = (-0.0001474*pow(self.T, 3) + \
                   0.348886926471821*pow(self.T, 2) \
                   - 277.603979928015*self.T + \
                   75514.7595133316) *1e-6 # kg/m/s
        self.kappa = 0.00019*self.T + 0.3911015 # kg*m/s^3/K
        self.nu = self.mu / self.rho
        self.alpha = self.kappa / (self.rho * self.Cp)
        self.Pr = self.nu / self.alpha
        if self.debug==True:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

class Grid:
    
    """ A cylindrical coordinate (theta, r) grid class """
    
    def __init__(self, nr=6, nt=61, rMin=0.5, rMax=0.7,
                 thetaMin=0, thetaMax=np.radians(180)):
        self.nr, self.nt = nr, nt
        self.a, self.b = rMin, rMax
        r = np.linspace(rMin, rMax, nr)
        theta = np.linspace(thetaMin, thetaMax, nt)
        self.r, self.theta = np.meshgrid(r, theta)
        self.dr = float(rMax-rMin)/(nr-1)
        dTheta = float(thetaMax-thetaMin)/(nt-1)
        # face surface (sf) areas:
        self.sfRmin = (np.ones(nt) * pi * rMin) / (nt - 1)
        self.sfRmin[0] *= 0.5; self.sfRmin[-1] *= 0.5
        self.sfRmax = (np.ones(nt) * pi * rMax) / (nt - 1)
        self.sfRmax[0] *= 0.5; self.sfRmax[-1] *= 0.5
        # create 'ghost' elements for symmetry BCs:
        theta = np.insert(theta, 0, thetaMin-dTheta)
        theta = np.append(theta, thetaMax+dTheta)
        self.meshR, self.meshTheta = np.meshgrid(r, theta)
        # create constants for use in iterations:
        self.twoDrR = 2 * self.dr * self.meshR[1:-1,1:-1]
        self.dr2, self.dTheta2 = self.dr**2, dTheta**2
        self.dTheta2R2 = self.dTheta2 * self.meshR[1:-1,1:-1]**2
        self.dnr = (2. / self.dr2 + 2. / self.dTheta2R2)
        # create 'mask array' for front-side collimated flux logic:
        self.cosTheta = np.cos(theta)
        self.sinTheta = np.sin(theta)
        self.tubeFront = np.ones(len(theta))
        self.tubeFront[self.cosTheta<0] = 0.0

class Solver:
    
    """  A Laplacian solver for steady-state conduction in cylinders 
         -- Gauss-Seidel iteration of T(r, theta) 
         -- bi-harmonic thermoelastic stress post-processing """

    # Constants:
    sigma = 5.67e-8    # Stefan-Boltzmann
    
    def __init__(self, grid, debug=False, it='inline', CG=8.5e5, 
                 k=20, T_int=723.15, h_int=10e3, U=4.0, R_f=0., A=0.968, 
                 epsilon=0.87, T_ext=293.15, h_ext=30., P_i=0e5, 
                 alpha=18.5e-6, E=165e9, nu=0.3, n=1, bend=False):
        self.debug = debug
        # Class constants and variables (default UNS S31600 @ 450degC):
        self.g = grid
        self.setIterator(it)
        self.CG = CG            # concentration (C) x solar constant (G)
        self.k = k              # thermal conductivity of tube
        self.T_int = T_int      # temperature of heat transfer fluid
        self.h_int = h_int      # constant int convection coefficient
        self.R_f = R_f          # internal fouling coefficient
        self.A = A              # tube external surface absorptance
        self.epsilon = epsilon  # tube external emmissivity
        self.T_ext = T_ext      # ambient temperature
        self.h_ext = h_ext      # ext convection coefficient (with wind)
        self.P_i = P_i          # internal pipe pressure
        self.alpha = alpha      # thermal expansion coefficienct of tube
        self.E = E              # Modulus of elasticity
        self.nu = nu            # Poisson's coefficient
        self.n = n              # Number of Fourier 'frequencies'
        self.bend = bend        # switch to allow tube bending
        self.meshT = np.ones((grid.nt+2, grid.nr), 'd') * T_int
        self.T = self.meshT[1:-1,:] # remove symm for post-processing

    def computeError(self):        
        """ Computes absolute error using an L2 norm for the solution.
        This requires that self.T and self.old_T must be appropriately
        setup - only used for numpyStep """        
        v = (self.meshT - self.old_T).flat
        return np.sqrt(np.dot(v,v))

    def numpyStep(self):
        """ Gauss-Seidel iteration using numpy expression. """
        self.old_T = self.meshT.copy()
        # "Driving" BCs (heat flux, radiation and convection)
        self.extBC()
        self.intBC()
        # Numpy iteration
        self.meshT[1:-1,1:-1] = ( 
            ( self.meshT[1:-1,2:] - self.meshT[1:-1,:-2] ) 
            / self.g.twoDrR +
            ( self.meshT[1:-1,:-2] + self.meshT[1:-1,2:] ) 
            / self.g.dr2 +
            ( self.meshT[:-2,1:-1] + self.meshT[2:,1:-1] ) 
            / self.g.dTheta2R2
        ) / self.g.dnr
        # Symmetry boundary conditions
        self.symmetryBC()
        return self.computeError()

    def blitzStep(self):
        """ Gauss-Seidel iteration using numpy expression
        that has been blitzed using weave """
        self.old_T = self.meshT.copy()
        # "Driving" BCs (heat flux, radiation and convection)
        self.extBC()
        self.intBC()
        # Prepare constants and arrays for blitz
        T = self.meshT
        twoDrR = self.g.twoDrR
        dr2 = self.g.dr2
        dTheta2R2 = self.g.dTheta2R2
        dnr = self.g.dnr
        expr = "T[1:-1,1:-1] = ("\
            "( T[1:-1,2:] - T[1:-1,:-2] ) / twoDrR +"\
            "( T[1:-1,:-2] + T[1:-1,2:] ) / dr2 +"\
            "( T[:-2,1:-1] + T[2:,1:-1] ) / dTheta2R2"\
            ") / dnr"
        weave.blitz(expr, check_size=0)
        # Transfer result back to mesh/grid
        self.meshT = T
        # Symmetry boundary conditions
        self.symmetryBC()
        return self.computeError()

    def inlineStep(self):
        """ Gauss-Seidel iteration using an inline C code """
        # "Driving" BCs (heat flux, radiation and convection)
        self.extBC()
        self.intBC()
        # Prepare constants and arrays for blitz
        T = self.meshT
        nt, nr = self.meshT.shape
        twoDrR = self.g.twoDrR
        dr2 = self.g.dr2
        dTheta2R2 = self.g.dTheta2R2
        dnr = self.g.dnr
        code = """
               #line 000 "laplacianCylinder.py"
               double tmp, err, diff;
               err = 0.0;
               for (int i=1; i<nt-1; ++i) {
                   for (int j=1; j<nr-1; ++j) {
                       tmp = T(i,j);
                       T(i,j) = ((T(i,j+1) - T(i,j-1))/twoDrR(i-1,j-1) +
                                 (T(i,j-1) + T(i,j+1))/dr2 +
                                 (T(i-1,j) + T(i+1,j))/dTheta2R2(i-1,j-1)
                                ) / dnr(i-1,j-1);
                       diff = T(i,j) - tmp;
                       err += diff*diff;
                   }
               }
               return_val = sqrt(err);
               """
        err = weave.inline(code,
                           ['nr', 'nt', 'T', 'twoDrR', 
                            'dr2', 'dTheta2R2', 'dnr'],
                           type_converters=converters.blitz,
                           compiler = 'gcc')
        # Transfer result back to mesh/grid
        self.meshT = T
        # Symmetry boundary conditions
        self.symmetryBC()
        return err

    def setIterator(self, iterator='numpy'):        
        """ Sets the iteration scheme to be used while solving given a
        string which should be one of ['numpy', 'blitz', 'inline']. """        
        if iterator == 'numpy':
            self.iterate = self.numpyStep
        elif iterator == 'blitz':
            self.iterate = self.blitzStep
        elif iterator == 'inline':
            self.iterate = self.inlineStep
        else:
            self.iterate = self.numpyStep            
                
    def solve(self, n_iter=0, eps=1.0e-16):        
        """ Solves the equation given:
        - an error precision -- eps
        - a maximum number of iterations -- n_iter """
        err = self.iterate()
        count = 1
        while err > eps:
            if n_iter and count >= n_iter:
                return err
            err = self.iterate()
            count = count + 1
        self.T = self.meshT[1:-1,:]
        return count

    def postProcessing(self):
        self.stress()
        return

    """ _____________________ BOUNDARY CONDITIONS _____________________ """

    def symmetryBC(self):        
        """ Sets the left and right symmetry BCs """       
        self.meshT[0, 1:-1] = self.meshT[2, 1:-1]
        self.meshT[-1, 1:-1] = self.meshT[-3, 1:-1]

    def tubeExtTemp(self):
        """ fixedValue boundary condition """        
        self.meshT[:,-1] = self.T_ext

    def tubeExtConv(self):
        """ Convective boundary condition """        
        self.meshT[:, -1] = (self.meshT[:,-2] + \
                             ((self.g.dr * self.h_ext / 
                               self.k) * self.T_ext)) \
            / (1 + (self.g.dr * self.h_ext / self.k))

    def tubeExtFlux(self):        
        """ Heat flux boundary condition """
        self.meshT[:,-1] = ((self.g.dr * self.CG) / 
                            self.k) + self.meshT[:, -2]

    def tubeExtCosFlux(self): 
        """ 100% absorbed cosine flux boundary condition """
        self.heatFluxInc = (self.g.tubeFront * \
                            self.CG * self.g.cosTheta)
        heatFluxAbs = self.heatFluxInc
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (heatFluxAbs * self.g.dr / self.k)

    def tubeExtCosFluxRadConv(self): 
        """ Heat flux, re-radiation and convection boundary condition """
        self.heatFluxInc = (self.g.tubeFront * \
                            self.CG * self.g.cosTheta)
        heatFluxAbs = self.heatFluxInc * self.A \
                      - (self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (heatFluxAbs * self.g.dr / self.k)

    def tubeExtCosFluxRadConvAdiabaticBack(self): 
        """ Heat flux, re-radiation and convection boundary condition """
        self.heatFluxInc = (self.g.tubeFront * \
                            self.CG * self.g.cosTheta)
        heatFluxAbs = self.heatFluxInc * self.A \
                      - (self.g.tubeFront * self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * self.g.tubeFront * 
                         (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (heatFluxAbs * self.g.dr / self.k)

    def tubeIntTemp(self):
        """ fixedValue boundary condition """        
        self.meshT[:,0] = self.T_int

    def tubeIntFlux(self):        
        """ Heat flux boundary condition """        
        self.meshT[:,0] = ((self.g.dr * self.CG) / 
                           self.k) + self.meshT[:, 1]

    def tubeIntConv(self):
        """ Convective boundary condition to tube flow with fouling """
        U = 1 / (self.R_f + (1 / self.h_int))
        self.meshT[:,0] = (self.meshT[:,1] + \
                           ((self.g.dr * U / self.k) * \
                            self.T_int)) \
            / (1 + (self.g.dr * U / self.k))

    """ _______________________ POST-PROCESSING _______________________ """

    def stress(self):
        """ The Timoshenko & Goodier approach (dating back to 1937):
        -- S. Timoshenko and J. N. Goodier. Theory of Elasticity. 
           p432, 1951.
        -- J. N. Goodier, Thermal Stresses and Deformation, 
           J. Applied Mechanics, Trans ASME, vol. 24(3), 
           p467-474, 1957. """
        # smaller names for equations below:
        a, b = self.g.a, self.g.b
        P_i, alpha, E, nu = self.P_i, self.alpha, self.E, self.nu
        # create a local 'harmonic' cylinder of T, theta and r:
        T = np.insert(self.T, 0, self.T[1:,:][::-1], axis=0)
        theta = np.linspace(np.radians(-180), np.radians(180), 
                            (self.g.nt*2)-1)
        r = np.linspace(a, b, self.g.nr)
        meshR, meshTheta = np.meshgrid(r, theta)
        # local time-saving variables:
        meshR2 = meshR**2; meshR4 = meshR**4
        a2 = a**2; a4 = a**4
        b2 = b**2; b4 = b**4
        # 'guess' of coefficients for curve_fit function:
        p0 = [1.0] * (1 + (s.n * 2))
        # inside:
        popt1, pcov1 = opt.curve_fit(fourierTheta, theta, T[:,0], p0)
        Tbar_i = popt1[0]; BP = popt1[1]; DP = popt1[2];
        # outside:
        popt2, pcov2 = opt.curve_fit(fourierTheta, theta, T[:,-1], p0)
        Tbar_o = popt2[0]; BPP = popt2[1]; DPP = popt2[2];
        kappa_theta = (( (((BP * b) - (BPP * a)) / (b2 + a2)) 
                         * np.cos(meshTheta)) + \
                       ( (((DP * b) - (DPP * a)) / (b2 + a2)) 
                         * np.sin(meshTheta))) * \
                         (meshR * a * b) / (b2 - a2)
        kappa_tau = (( (((BP * b) - (BPP * a)) / (b2 + a2)) 
                       * np.sin(meshTheta)) - \
                     ( (((DP * b) - (DPP * a)) / (b2 + a2)) 
                       * np.cos(meshTheta))) * \
                       (meshR * a * b) / (b2 - a2)
        if self.bend:
            kappa_noM = meshR * ((((BP * a) + (BPP * b)) / (b2 + a2) * \
                                  np.cos(meshTheta)) \
                                 + (((DP * a) + (DPP * b)) / (b2 + a2) * \
                                    np.sin(meshTheta)))
        else: kappa_noM = 0.0
        C = (alpha * E) / (2 * (1 - nu))
        # Axisymmetrical thermal stress component:
        kappa = (Tbar_i - Tbar_o) / np.log(b/a)
        QR = kappa * C * (- np.log(b/meshR) - \
                          (a2/(b2 - a2) * (1 - b2/meshR2) * np.log(b/a)))
        QTheta = kappa * C * (1 - np.log(b/meshR) - \
                              (a2/(b2 - a2) * (1 + b2/meshR2) * np.log(b/a)))
        QZ = kappa * C * (1 - (2*np.log(b/meshR)) - (2 * a2 / (b2 - a2) * \
                                                     np.log(b/a)))
        # Nonaxisymmetrical T:
        T_theta = T - ((Tbar_i - Tbar_o) * \
                       np.log(b / meshR) / np.log(b / a)) - Tbar_o
        self.T_theta = T_theta
        # Nonaxisymmetric thermal stress component:
        QR += C * kappa_theta * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
        QTheta += C * kappa_theta * (3 - ((a2 + b2) / meshR2) - \
                                     ((a2 * b2) / meshR4))
        QZ += alpha * E * ((kappa_theta * (nu / (1 - nu)) * \
                            (2 - ((a2 + b2) / meshR2))) + \
                           kappa_noM - T_theta)
        QRTheta = C * kappa_tau * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
        QEq = np.sqrt(0.5 * ((QR - QTheta)**2 + \
                             (QTheta - QZ)**2 + \
                             (QZ - QR)**2) + \
                      6 * (QRTheta**2))
        # Pressure stress component:
        PR = ((a2 * self.P_i) / (b2 - a2)) * (1 - (b2 / meshR2))
        PTheta = ((a2 * self.P_i) / (b2 - a2)) * (1 + (b2 / meshR2))
        PZ = 0 #(a2 * self.P_i) / (b2 - a2)
        PEq = np.sqrt(0.5 * ((PR - PTheta)**2 + \
                             (PTheta - PZ)**2 + \
                             (PZ - PR)**2))
        sigmaR = QR + PR
        sigmaTheta = QTheta + PTheta
        sigmaZ = QZ + PZ
        sigmaRTheta = QRTheta
        # Equivalent/vM stress:
        sigmaEq = np.sqrt(0.5 * ((sigmaR - sigmaTheta)**2 + \
                                 (sigmaTheta - sigmaZ)**2 + \
                                 (sigmaZ - sigmaR)**2) + \
                          6 * (sigmaRTheta**2))
        self.popt1 = popt1
        self.popt2 = popt2
        self.sigmaR = sigmaR[self.g.nt-1:,:]
        self.sigmaRTheta = sigmaRTheta[self.g.nt-1:,:]
        self.sigmaTheta = sigmaTheta[self.g.nt-1:,:]
        self.sigmaZ = sigmaZ[self.g.nt-1:,:]
        self.sigmaEq = sigmaEq[self.g.nt-1:,:]
        if self.debug:
            valprint('Tbar_i', Tbar_i, 'K')
            valprint('B\'_1', BP, 'K')
            valprint('D\'_1', DP, 'K')
            valprint('Tbar_o', Tbar_o, 'K')
            valprint('B\'\'_1', BPP, 'K')
            valprint('D\'\'_1', DPP, 'K')
            valprint('sigma_r', self.sigmaR[0,-1]*1e-6, 'MPa')
            valprint('sigma_rTheta', self.sigmaRTheta[0,-1]*1e-6, 'MPa')
            valprint('sigma_theta', self.sigmaTheta[0,-1]*1e-6, 'MPa')
            valprint('sigma_z', self.sigmaZ[0,-1]*1e-6, 'MPa')
            valprint('max(sigma_Eq)', np.max(self.sigmaEq)*1e-6, 'MPa')

""" __________________________ USAGE (MAIN) ___________________________ """

if __name__ == "__main__":

    """ NPS Sch. 5S 1" SS316 at 450degC """
    a = 30.098/2e3     # inside tube radius [mm->m]
    b = 33.4/2e3       # outside tube radius [mm->m]

    """ Create instance of Grid: """
    g = Grid(nr=12, nt=91, rMin=a, rMax=b) # nr, nt -> resolution

    """ Create instance of LaplaceSolver: """
    s = Solver(g, debug=True, CG=0.85e6, k=20, T_int=723.15, R_f=0,
               A=0.968, epsilon=0.87, T_ext=293.15, h_ext=20., 
               P_i=0e5, alpha=18.5e-6, E=165e9, nu=0.31, n=1,
               bend=False)

    """ Any of the properties defined above can be changed, e.g.: """
    # s.CG = 1.2e5 ...

    """ External BC: """
    #s.extBC = s.tubeExtTemp
    #s.extBC = s.tubeExtFlux
    #s.extBC = s.tubeExtConv
    #s.extBC = s.tubeExtCosFluxRadConv
    s.extBC = s.tubeExtCosFluxRadConvAdiabaticBack

    """ Internal BC: """
    #s.intBC = s.tubeIntTemp
    #s.intBC = s.tubeIntFlux
    s.intBC = s.tubeIntConv

    headerprint(' HTC : 10e3 W/m^s/K ')
    s.h_int = 10e3
    t = time.clock(); ret = s.solve(eps=1e-6); s.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    """ To access the temperature distribution: """
    #     s.T[theta,radius] using indexes set by nr and nt
    """ e.g. s.T[0,-1] is outer tube front """

    """ Same goes for stress fields: """
    #     s.sigmaR[theta,radius]
    #     s.sigmaTheta[theta,radius]
    #     s.sigmaZ[theta,radius]
    #     s.sigmaEq[theta,radius]

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'htc10_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'htc10_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'htc10_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'htc10_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'htc10_sigmaEq.pdf')

    headerprint(' HTC : 40e3 W/m^s/K ')
    s.h_int = 40e3
    t = time.clock(); ret = s.solve(eps=1e-6); s.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'htc40_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'htc40_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'htc40_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'htc40_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'htc40_sigmaEq.pdf')


    headerprint(' ASTRI 2.0 REFERENCE CASE ')
    headerprint('Inco625 at 650 degC', ' ')
    b = 25.4e-3/2.     # inside tube radius (mm->m)
    valprint('b', b*1e3, 'mm')
    a = b - 1.65e-3    # outside tube radius (mm->m)
    valprint('a', a*1e3, 'mm')
    k = 19.15          # thermal conductivity (kg*m/s^3/K)
    valprint('k', k, 'kg*m/s^3/K')
    alpha = 18.815e-6  # thermal dilation (K^-1)
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')
    E = 168e9          # Youngs modulus (Pa)
    valprint('E', E*1e-9, 'GPa')
    nu = 0.31          # Poisson
    valprint('nu', nu)
    CG = 7.5e5        # absorbed flux (W/m^2)
    valprint('CG', CG*1e-3, 'kW/m^2')
    mdot = 0.2         # mass flow (kg/s)
    valprint('mdot', mdot, 'kg/s')
    T_int = 887        # bulk sodium temperature (K)
    
    """ Create instance of Grid: """
    g = Grid(nr=12, nt=91, rMin=a, rMax=b) # nr, nt -> resolution

    """ Create instance of LaplaceSolver: """
    s = Solver(g, debug=True, CG=CG, k=k, T_int=T_int, R_f=0,
               P_i=0e5, alpha=alpha, E=E, nu=nu, n=1,
               bend=False)

    """ External BC: """
    #s.extBC = s.tubeExtTemp
    s.extBC = s.tubeExtCosFlux
    #s.extBC = s.tubeExtConv
    #s.extBC = s.tubeExtCosFluxRadConv
    #s.extBC = s.tubeExtCosFluxRadConvAdiabaticBack

    """ Internal BC: """
    #s.intBC = s.tubeIntTemp
    #s.intBC = s.tubeIntFlux
    s.intBC = s.tubeIntConv
    sodium = liquidSodium(True); sodium.update(T_int)
    #s.h_int = HTC(True, sodium, a, b, s.k, 'velocity', 4.0)
    s.h_int = HTC(True, sodium, a, b, s.k, 'mdot', mdot)
    #s.h_int = HTC(True, sodium, a, b, s.k, 'heatCapRate', 5000)

    t = time.clock(); ret = s.solve(eps=1e-6)
    headerprint('Analytical thermoelastic stress', ' ')
    s.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'Inco625_liquidSodium_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'Inco625_liquidSodium_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'Inco625_liquidSodium_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'Inco625_liquidSodium_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'Inco625_liquidSodium_sigmaEq.pdf')
