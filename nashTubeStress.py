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
from math import exp, log, sqrt, pi, ceil, floor, asin
import numpy as np # version 1.15.4
from numpy import ma
#import weave # version 0.17.0
#from weave import converters
import scipy.optimize as opt # version 1.1.0
from pint import UnitRegistry # version 0.8.1
UR_ = UnitRegistry()
Q_ = UR_.Quantity

# Plotting:
import matplotlib as mpl # version 2.2.3
import matplotlib.pyplot as plt
## uncomment following if not in ~/.config/matplotlib/matplotlibrc
#params = {'text.latex.preamble': [r'\usepackage{newtxtext,newtxmath,siunitx}']}
#plt.rcParams.update(params)
mpl.rc('figure.subplot', bottom=0.13, top=0.95)
mpl.rc('figure.subplot', left=0.15, right=0.95)
from matplotlib import colors, ticker, cm
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.projections import PolarAxes
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from mpl_toolkits.axisartist.grid_finder import \
    (FixedLocator, MaxNLocator, DictFormatter)
# if you're matplotlib is older than version 2:
#import colormaps as cmaps # magma, inferno, plasma, viridis

#################################### CLASSES ###################################

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

    def __init__(self, grid, debug=False, it='numpy', CG=8.5e5,
                 k=20, T_int=723.15, h_int=10e3, U=4.0, R_f=0., A=0.968,
                 epsilon=0.87, T_ext=293.15, h_ext=30., P_i=0e5,
                 T_0=0., alpha=18.5e-6, E=165e9, nu=0.3, n=1,
                 bend=False, GPS=True, wind_speed=7.5, wind_dir=90., coating='CoteRill750',
                 view_factor=1.0, internal_flux=15.23e3):
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
        self.T_0 = T_0          # stress free temperature
        self.alpha = alpha      # thermal expansion coefficienct of tube
        self.E = E              # Modulus of elasticity
        self.nu = nu            # Poisson's coefficient
        self.n = n              # Number of Fourier 'frequencies'
        self.bend = bend        # switch to allow tube bending
        self.GPS = GPS          # switch to turn off generalised plane strain
        self.meshT = np.ones((grid.nt+2, grid.nr), 'd') * T_int
        self.T = self.meshT[1:-1,:]  # remove symm for post-processing
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        self.coating = coating
        self.view_factor = view_factor  # Radiative view factor between tube and environment
        self.int_flux = internal_flux  # internal flux from hotbox (W/m2)

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
        pass
# =============================================================================
#         """ Gauss-Seidel iteration using numpy expression
#         that has been blitzed using weave """
#         self.old_T = self.meshT.copy()
#         # "Driving" BCs (heat flux, radiation and convection)
#         self.extBC()
#         self.intBC()
#         # Prepare constants and arrays for blitz
#         T = self.meshT
#         twoDrR = self.g.twoDrR
#         dr2 = self.g.dr2
#         dTheta2R2 = self.g.dTheta2R2
#         dnr = self.g.dnr
#         expr = "T[1:-1,1:-1] = ("\
#             "( T[1:-1,2:] - T[1:-1,:-2] ) / twoDrR +"\
#             "( T[1:-1,:-2] + T[1:-1,2:] ) / dr2 +"\
#             "( T[:-2,1:-1] + T[2:,1:-1] ) / dTheta2R2"\
#             ") / dnr"
#         weave.blitz(expr, check_size=0)
#         # Transfer result back to mesh/grid
#         self.meshT = T
#         # Symmetry boundary conditions
#         self.symmetryBC()
#         return self.computeError()
# =============================================================================

    def inlineStep(self):
        pass
# =============================================================================
#         """ Gauss-Seidel iteration using an inline C code """
#         # "Driving" BCs (heat flux, radiation and convection)
#         self.extBC()
#         self.intBC()
#         # Prepare constants and arrays for blitz
#         T = self.meshT
#         nt, nr = self.meshT.shape
#         twoDrR = self.g.twoDrR
#         dr2 = self.g.dr2
#         dTheta2R2 = self.g.dTheta2R2
#         dnr = self.g.dnr
#         code = """
#                #line 000 "nashTubeStress.py"
#                double tmp, err, diff;
#                err = 0.0;
#                for (int i=1; i<nt-1; ++i) {
#                    for (int j=1; j<nr-1; ++j) {
#                        tmp = T(i,j);
#                        T(i,j) = ((T(i,j+1) - T(i,j-1))/twoDrR(i-1,j-1) +
#                                  (T(i,j-1) + T(i,j+1))/dr2 +
#                                  (T(i-1,j) + T(i+1,j))/dTheta2R2(i-1,j-1)
#                                 ) / dnr(i-1,j-1);
#                        diff = T(i,j) - tmp;
#                        err += diff*diff;
#                    }
#                }
#                return_val = sqrt(err);
#                """
#         err = weave.inline(code,
#                            ['nr', 'nt', 'T', 'twoDrR',
#                             'dr2', 'dTheta2R2', 'dnr'],
#                            type_converters=converters.blitz,
#                            compiler = 'gcc')
#         # Transfer result back to mesh/grid
#         self.meshT = T
#         # Symmetry boundary conditions
#         self.symmetryBC()
#         return err
# =============================================================================

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

    ############################ BOUNDARY CONDITIONS ###########################

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


    def extHTC(self, temp, wind_speed=7.5, wind_dir=90):
        """
        external heat transfer coefficients due to convection based on Vast Solar receiver geometry
        as of 26/04/2020. Includes a cowling and roof. These linear correlations based on external
        CFD parametric study conducted by Politecnico di Torino.

        wind_speed in m/s
        wind_dir in degree from front_on to receiver
        temp (external surface temperature) in degrees Kelvin
        """

        if wind_speed == 0:
            return 0.000884468*temp + 3.024050054

        elif wind_speed == 7.5:
            if wind_dir == 0:
                return 0.000298186*temp + 4.147614501
            elif wind_dir == 90:
                return -0.002313861*temp + 9.89913916
            elif wind_dir == 180:
                return -0.003201793*temp + 11.88847284
            else:
                raise ValueError("Error: incorrect wind_dir value entered")

        elif wind_speed == 20.0:
            if wind_dir == 0:
                return -0.002535205*temp + 8.908062072
            elif wind_dir == 90:
                return -0.008124657*temp + 25.72204594
            elif wind_dir == 180:
                return -0.005727515*temp + 25.04489471
            else:
                raise ValueError("Error: incorrect wind_dir value entered")

        # TODO complete other angles
        else:
            raise ValueError("Error: incorrect wind_speed value entered")

    def extVariableEmissivity(self, temp):
        temp = temp - 273.15  # From Kelvin to Celcius
        if self.coating == 'Pyromark':
            '''C. Ho et al., Journal of Solar Energy Engineering 136 (2014)'''
            epsilon = -3.e-13*temp**4 + 9.0e-10*temp**3 - 1.0E-6*temp**2 + 0.0006*temp + 0.7441
            return epsilon
        elif self.coating == 'CoteRill750':
            epsilon = -1.5716e-10*temp**3 + 3.1795e-7*temp**2 - 0.0001833*temp + 0.946911
            return epsilon
        else:
            raise ValueError("Error: incorrect coating entered")

    def extTubeHalfCosFluxVarRadVarConvAdiabatcBack(self):
        """ Heat flux, re-radiation as a function of temperature, convection as a function of
        temperature boundary condition"""
        self.phi_inc = (self.g.halfTube *
                        self.CG * self.g.cosTheta)
        h_ext = self.g.halfTube * self.extHTC(self.meshT[:, -1], self.wind_speed, self.wind_dir)

        if self.epsilon == 'variable':
            epsilon = self.extVariableEmissivity(self.meshT[:, -1])
        else:
            Warning('Variable emissivity boundary condition selected but constant epsilon given.'
                    'Using constant value')
            epsilon = self.epsilon

        phi_t = self.phi_inc * self.A \
                - (self.g.halfTube * self.view_factor * self.sigma * epsilon \
                         * (self.meshT[:, -1]**4 - self.T_ext**4)) \
                      - (h_ext * self.g.halfTube *
                         (self.meshT[:, -1] - self.T_ext))
        self.meshT[:, -1] = self.meshT[:, -2] + \
                               (phi_t * self.g.dr / self.k)

    def naturalConvectionSiebers(self):
        ''' Calculates the natural convection coefficient Properties Calculated at T_ext'''

        # Air Properties
        density = 358.517*self.T_ext**-1.00212
        volume_exp_coeff = 1.00212/self.T_ext
        conductivity = -1.3707e-8*self.T_ext**2+7.616e-5*self.T_ext+4.5968e-3
        kinematic_viscosity = (-8.3123e-12*self.T_ext**2+4.4156e-8*self.T_ext+6.2299e-6)/density

        internal_wall_av_T = self.meshT[self.g.halfTube == 0.0, -1].mean()

        h_int = conductivity * 0.098 * (9.81*volume_exp_coeff*(internal_wall_av_T - self.T_ext) /
                                        kinematic_viscosity**2)**(1./3.)
        return h_int

    def extTubeHalfCosFluxVarRadVarConvNatConvBack(self):
        """ Heat flux, re-radiation as a function of temperature, convection as a function of
        temperature boundary condition"""
        self.phi_inc = (self.g.halfTube *
                        self.CG * self.g.cosTheta)
        h_ext = self.g.halfTube * self.extHTC(self.meshT[:, -1], self.wind_speed, self.wind_dir)
        h_int = self.naturalConvectionSiebers()

        if self.epsilon == 'variable':
            epsilon = self.extVariableEmissivity(self.meshT[:, -1])
        else:
            Warning('Variable emissivity boundary condition selected but constant epsilon given.'
                    'Using constant value')
            epsilon = self.epsilon

        phi_t = self.phi_inc * self.A \
                      - (self.g.halfTube * self.view_factor * self.sigma * epsilon \
                         * (self.meshT[:, -1]**4 - self.T_ext**4)) \
                      - (h_ext * self.g.halfTube *
                         (self.meshT[:, -1] - self.T_ext)) \
                      - (h_int * (1. - self.g.halfTube) *
                         (self.meshT[:, -1] - self.T_ext))
        self.meshT[:, -1] = self.meshT[:, -2] + \
                           (phi_t * self.g.dr / self.k)

    def extTubeHalfCosFluxVarRadVarConvReflFluxBack(self):
        """ Heat flux, re-radiation as a function of temperature, convection as a function of
        temperature boundary condition, rearside flux due to hotbox"""
        self.phi_inc = (self.g.halfTube *
                        self.CG * self.g.cosTheta)
        h_ext = self.g.halfTube * self.extHTC(self.meshT[:, -1], self.wind_speed, self.wind_dir)

        if self.epsilon == 'variable':
            epsilon = self.extVariableEmissivity(self.meshT[:, -1])
        else:
            Warning('Variable emissivity boundary condition selected but constant epsilon given.'
                    'Using constant value')
            epsilon = self.epsilon

        phi_t = self.phi_inc * self.A \
                      - (self.g.halfTube * self.view_factor * self.sigma * epsilon \
                         * (self.meshT[:, -1]**4 - self.T_ext**4)) \
                      - (h_ext * self.g.halfTube *
                         (self.meshT[:, -1] - self.T_ext)) \
                      + self.int_flux * -self.g.cosTheta * (1. - self.g.halfTube)

        self.meshT[:, -1] = self.meshT[:, -2] + \
                           (phi_t * self.g.dr / self.k)

    def intTubeTemp(self):
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

    ############################## POST-PROCESSING #############################

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
        p0 = [1.0] * (1 + (self.n * 2))
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
        # Plane strain override (needs T_0 defined):
        if not self.GPS:
            QZ = (nu*(QR + QTheta)) - (E*alpha*(T-self.T_0))
        QRTheta = C * kappa_tau * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
        QEq = np.sqrt(0.5 * ((QR - QTheta)**2 + \
                             (QTheta - QZ)**2 + \
                             (QZ - QR)**2) + \
                      6 * (QRTheta**2))
        # Pressure stress component:
        PR = ((a2 * self.P_i) / (b2 - a2)) * (1 - (b2 / meshR2))
        PTheta = ((a2 * self.P_i) / (b2 - a2)) * (1 + (b2 / meshR2))
        ## uncomment for simple plane strain:
        PZ = 0. ##(a2 * self.P_i) / (b2 - a2)
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
            headerprint('Stress at outside tube crown:', ' ')
            valprint('sigma_r', self.sigmaR[0,-1]*1e-6, 'MPa')
            valprint('sigma_rTheta', self.sigmaRTheta[0,-1]*1e-6, 'MPa')
            valprint('sigma_theta', self.sigmaTheta[0,-1]*1e-6, 'MPa')
            valprint('sigma_z', self.sigmaZ[0,-1]*1e-6, 'MPa')
            valprint('sigma_Eq', self.sigmaEq[0,-1]*1e-6, 'MPa')

################################### PLOTTING ###################################

def plotTemperatureAnnotate(theta, r, T, TMin, TMax, filename):
    fig = plt.figure(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=-1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    #cmap = cmaps.magma # magma, inferno, plasma, viridis...
    cmap = cm.get_cmap('magma')
    levels = ticker.MaxNLocator(nbins=10).tick_values(TMin-273.15, TMax-273.15)
    cf = ax.contourf(theta, r, T-273.15, levels=levels, cmap=cmap)
    ax.set_rmin(0)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r'\textsc{temperature}, $T$ (\si{\celsius})')
    ax.patch.set_visible(False)
    ax.spines['polar'].set_visible(False)
    gridlines = ax.get_xgridlines()
    ticklabels = ax.get_xticklabels()
    for i in range(5, len(gridlines)):
        gridlines[i].set_visible(False)
        ticklabels[i].set_visible(False)
    ax.annotate(r'${0:.0f}$'.format(T.max()-273.15)+' (\si{\celsius})', \
                 xy=(theta[0,-1], r[0,-1]), \
                 xycoords='data', xytext=(40, 10), \
                 textcoords='offset points', fontsize=12, \
                 arrowprops=dict(arrowstyle='->'))
    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x', linewidth=0.2)
    plt.setp(ax.get_yticklabels(), visible=False)
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotStress(theta, r, sigma, sigmaMin, sigmaMax, filename):
    fig = plt.figure(figsize=(3.5, 3.5))
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
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotStressAnnotate(theta, r, sigma, sigmaMin, sigmaMax, annSide, filename):
    annSide = -70 if annSide=='left' else 40
    fig = plt.figure(figsize=(3, 3.25))
    fig.subplots_adjust(left=-1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    cmap = cm.get_cmap('magma')
    #sigmaMin = np.min(sigma*1e-6); sigmaMax = np.max(sigma*1e-6)
    levels = ticker.MaxNLocator(nbins=10).tick_values(sigmaMin*1e-6, sigmaMax*1e-6)
    cf = ax.contourf(theta, r, sigma*1e-6, levels=levels, cmap=cmap)
    ax.set_rmin(0)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r'\textsc{equivalent stress}, $\sigma_\mathrm{eq}$ (MPa)')
    ax.patch.set_visible(False)
    ax.spines['polar'].set_visible(False)
    gridlines = ax.get_xgridlines()
    ticklabels = ax.get_xticklabels()
    for i in range(5, len(gridlines)):
        gridlines[i].set_visible(False)
        ticklabels[i].set_visible(False)
    #annInd = np.unravel_index(s.sigmaEq.argmax(), s.sigmaEq.shape)
    annInd = (0, -1)
    ax.annotate('${0:.0f}$ (MPa)'.format(np.max(sigma*1e-6)), \
                 xy=(theta[annInd], r[annInd]), \
                 xycoords='data', xytext=(annSide, 10), \
                 textcoords='offset points', fontsize=12, \
                 arrowprops=dict(arrowstyle='->'))
    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x', linewidth=0.2)
    plt.setp(ax.get_yticklabels(), visible=False)
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotComponentStress(r, sigmaR, sigmaTheta, sigmaZ,
                        sigmaEq, fea, filename, i, loc):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(r[i,:]*1e3, sigmaR[i,:]*1e-6, #'b-',
            label='$\sigma_r$')
    ax.plot(r[i,:]*1e3, sigmaTheta[i,:]*1e-6, #'c-',
            label=r'$\sigma_\theta$')
    ax.plot(r[i,:]*1e3, sigmaZ[i,:]*1e-6, #'m-',
            label='$\sigma_z$')
    ax.plot(r[i,:]*1e3, sigmaEq[i,:]*1e-6, #'k-',
            label='$\sigma_\mathrm{Eq}$')
    ax.legend(loc=loc)
    ax.set_xlabel('$r$ (mm)')
    ax.set_xlim((a*1e3)-0.1,(b*1e3)+0.1)
    ax.set_ylabel('$\sigma$ (MPa)')
    #labels = ax.get_xticklabels()
    #plt.setp(labels, rotation=30)
    fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotTIMO(r, s, feaCmp, feaEq, filename):
    a = r[0,0]; b = r[0,-1]
    trX = Q_(1, 'inch').to('mm').magnitude
    trY = Q_(1, 'ksi').to('MPa').magnitude
    trans = mtransforms.Affine2D().scale(trX,trY)
    fig = plt.figure(figsize=(4, 3.5))
    ax = SubplotHost(fig, 1, 1, 1)
    axa = ax.twin(trans)
    axa.set_viewlim_mode("transform")
    axa.axis["top"].set_label(r'\textsc{radius}, $r$ (in.)')
    axa.axis["top"].label.set_visible(True)
    axa.axis["right"].set_label(r'\textsc{stress component}, $\sigma$ (ksi)')
    axa.axis["right"].label.set_visible(True)
    ax = fig.add_subplot(ax)
    ax.plot(r[0,:]*1e3, s.sigmaTheta[0,:]*1e-6, '-', color='C0')
    ax.plot((a+feaCmp[:,0])*1e3, feaCmp[:,4]*1e-6, 'o', color='C0')
    ax.plot(r[0,:]*1e3, s.sigmaR[0,:]*1e-6, '-', color='C1')
    ax.plot((a+feaCmp[:,0])*1e3, feaCmp[:,5]*1e-6, '^', color='C1')
    ax.plot(r[0,:]*1e3, s.sigmaZ[0,:]*1e-6, '-', color='C2')
    ax.plot((a+feaCmp[:,0])*1e3, feaCmp[:,6]*1e-6, 'v', color='C2')
    ax.plot(r[0,:]*1e3, s.sigmaEq[0,:]*1e-6, '-', color='C3')
    ax.plot((a+feaEq[:,0])*1e3, feaEq[:,1]*1e-6, 's', color='C3')
    ax.plot(r[0,:]*1e3, s.sigmaRTheta[0,:]*1e-6, '-', color='C4')
    ax.plot((a+feaCmp[:,0])*1e3, feaCmp[:,7]*1e-6, '+', color='C4')
    ax.set_xlabel(r'\textsc{radius}, $r$ (mm)')
    ax.set_xlim((a*1e3)-10,(b*1e3)+10)
    ax.set_ylabel(r'\textsc{stress component}, $\sigma$ (MPa)')
    #ax.set_ylim(-400, 400)
    c0line = Line2D([], [], color='C0', marker='o',
                    label=r'$\sigma_\theta$')
    c1line = Line2D([], [], color='C1', marker='^',
                    label=r'$\sigma_r$')
    c2line = Line2D([], [], color='C2', marker='v',
                    label=r'$\sigma_z$')
    c3line = Line2D([], [], color='C3', marker='s',
                    label=r'$\sigma_\mathrm{eq}$')
    c4line = Line2D([], [], color='C4', marker='+',
                    label=r'$\tau_{r\theta}$')
    handles=[c0line, c1line, c2line, c4line, c3line]
    labels = [h.get_label() for h in handles]
    ax.legend([handle for i,handle in enumerate(handles)],
              [label for i,label in enumerate(labels)], loc='best')
    fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotNACA(r, sigma, fea, i, filename, loc, ylabel):
    a = r[0,0]; b = r[0,-1]
    trX = Q_(1, 'inch').to('mm').magnitude
    trY = Q_(1, 'ksi').to('MPa').magnitude
    trans = mtransforms.Affine2D().scale(trX,trY)
    fig = plt.figure(figsize=(4, 3.5))
    ax = SubplotHost(fig, 1, 1, 1)
    axa = ax.twin(trans)
    axa.set_viewlim_mode("transform")
    axa.axis["top"].set_label(r'\textsc{radius}, $r$ (in.)')
    axa.axis["top"].label.set_visible(True)
    axa.axis["right"].set_label(ylabel+' (ksi)')
    axa.axis["right"].label.set_visible(True)
    ax = fig.add_subplot(ax)
    ax.plot(r[0,:]*1e3, sigma[0,:]*1e-6, '-',
            color='C0',label=r'$\theta=0^\circ$')
    ax.plot((a+fea[0][:,0])*1e3, fea[0][:,i]*1e-6, 'o',
            color='C0', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[20,:]*1e-6, '-',
            color='C1', label=r'$\theta=60^\circ$')
    ax.plot((a+fea[1][:,0])*1e3, fea[1][:,i]*1e-6, '^',
            color='C1', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[40,:]*1e-6, '-',
            color='C2', label=r'$\theta=120^\circ$')
    ax.plot((a+fea[2][:,0])*1e3, fea[2][:,i]*1e-6, 'v',
            color='C2', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[60,:]*1e-6, '-',
            color='C3', label=r'$\theta=180^\circ$')
    ax.plot((a+fea[3][:,0])*1e3, fea[3][:,i]*1e-6, 's',
            color='C3', markevery=1)
    ax.set_xlabel(r'\textsc{radius}, $r$ (mm)')
    ax.set_xlim((a*1e3)-10,(b*1e3)+10)
    ax.set_ylabel(ylabel+' (MPa)')
    #ax.set_ylim(-400, 400)
    c0line = Line2D([], [], color='C0', marker='o',
                   label=r'$\theta=0^\circ$')
    c1line = Line2D([], [], color='C1', marker='^',
                   label=r'$\theta=60^\circ$')
    c2line = Line2D([], [], color='C2', marker='v',
                   label=r'$\theta=120^\circ$')
    c3line = Line2D([], [], color='C3', marker='s',
                   label=r'$\theta=180^\circ$')
    handles=[c0line, c1line, c2line, c3line]
    labels = [h.get_label() for h in handles]
    ax.legend([handle for i,handle in enumerate(handles)],
              [label for i,label in enumerate(labels)], loc=loc)
    fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotASTER(r, sigma, fea, i, filename, loc, ylabel):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    a = r[0,0]
    ax.plot(r[0,:]*1e3, sigma[0,:]*1e-6, '-', color='C0',
            label=r'$\theta=0^\circ$')
    ax.plot((a+fea[0][:,0])*1e3, fea[0][:,i], 'o', color='C0', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[20,:]*1e-6, '-', color='C1',
            label=r'$\theta=60^\circ$')
    ax.plot((a+fea[1][:,0])*1e3, fea[1][:,i], '^', color='C1', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[40,:]*1e-6, '-', color='C2',
            label=r'$\theta=120^\circ$')
    ax.plot((a+fea[2][:,0])*1e3, fea[2][:,i], 'v', color='C2', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[60,:]*1e-6, '-', color='C3',
            label=r'$\theta=180^\circ$')
    ax.plot((a+fea[3][:,0])*1e3, fea[3][:,i], 's', color='C3', markevery=1)
    ax.set_xlabel('$r$ (mm)')
    #ax.set_xlim((a*1e3)-10,(b*1e3)+10)
    ax.set_ylabel(ylabel+' (MPa)')
    c0line = Line2D([], [], color='C0', marker='o',
                   label=r'$\theta=0^\circ$')
    c1line = Line2D([], [], color='C1', marker='^',
                   label=r'$\theta=60^\circ$')
    c2line = Line2D([], [], color='C2', marker='v',
                   label=r'$\theta=120^\circ$')
    c3line = Line2D([], [], color='C3', marker='s',
                   label=r'$\theta=180^\circ$')
    handles=[c0line, c1line, c2line, c3line]
    labels = [h.get_label() for h in handles]
    ax.legend([handle for i,handle in enumerate(handles)],
              [label for i,label in enumerate(labels)], loc=loc)
    fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

################################### FUNCTIONS ##################################

def fourierTheta(theta, a0, *c):
    """ Timoshenko & Goodier equation """
    ret = a0 + np.zeros(len(theta))
    for i, n in zip(range(0,len(c),2), range(1,int((len(c)/2))+1)):
        ret += (c[i] * np.cos(n * theta)) + (c[i+1] * np.sin(n * theta))
    return ret

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

def valeprint(string, value, unit='-'):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .4e} ({2})".format(string, value, unit))

def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)

def HTC(debug, thermo, a, b, k, correlation, mode, arg):
    """
    Inputs:
        debug : (default:False)
        thermo : liquidSodium, nitrateSalt, chlorideSalt
        a : tube inner diameter (m)
        b : tube outer diameter (m)
        k : tube thermal conductivity (W/(m.K))
        corrlation : 'Dittus', 'Skupinski', 'Sleicher', ...
        mode : 'velocity', 'mdot', 'heatCapRate' (m/s, kg/s, J/K/s)
        arg : either velocity, mass-flow or heat capacity rate
    Return:
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
    else: sys.exit('Mode: {} not recognised'.format(mode))
    Re = U * d_i / thermo.nu # Reynolds
    Pe = Re * thermo.Pr # Peclet
    f = pow(0.790 * np.log(Re) - 1.64, -2)
    DP_f = -f * (0.5 * thermo.rho * pow(U, 2)) \
              / d_i # kg/m/s^2 for a metre of pipe!
    # if isinstance(thermo, liquidSodium):
    if correlation == 'Dittus':
        # Dittus-Boelter (Holman p286):
        Nu = 0.023 * pow(Re, 0.8) * pow(thermo.Pr, 0.4)
    elif correlation == 'Skupinski':
        # # Skupinski, Tortel and Vautrey:
        # # https://doi.org/10.1016/0017-9310(65)90077-3
        Nu = 4.82 + 0.0185 * pow(Pe, 0.827)
    elif correlation == 'Notter':
        # Notter and Schleicher:
        # https://doi.org/10.1016/0009-2509(72)87065-9
        Nu = 6.3 + 0.0167 * pow(Pe, 0.85) * pow(thermo.Pr, 0.08)
    elif correlation == 'Chen':
        # Chen and Chiou:
        # https://doi.org/10.1016/0017-9310(81)90167-8
        Nu = 5.6 + 0.0165 * pow(Pe, 0.85) * pow(thermo.Pr, 0.01)
    else: sys.exit('Correlation: {} not recognised'.format(correlation))
    h = Nu * thermo.kappa / d_i
    Bi = (t * h) / k
    if debug==True:
        valprint('U', U, 'm/s')
        valprint('mdot', mdot, 'kg/s')
        valprint('Re', Re)
        valprint('Pe', Pe)
        valprint('deltaP', DP_f, 'Pa/m')
        valprint('HCR', hcr, 'J/K/s')
        valprint('h_int', h, 'W/m^2/K')
        valprint('Bi', Bi)
    return [h, DP_f]

def findFlux(flux, s, f, i, point):
    """
    Helper for finding optimum flux for certain stress condition
    """
    s.CG = flux

    ret = s.solve(eps=1e-6)
    s.postProcessing()

    if point=='max':
        # T_max|sigmaEqMax:
        sigmaEqMax = np.interp(np.max(s.T), f[:,0], f[:,i])
        return sigmaEqMax - np.max(s.sigmaEq)
    elif point=='inside':
        # T_i:
        sigmaEqMax = np.interp(s.T[0,0], f[:,0], f[:,i])
        return sigmaEqMax - s.sigmaEq[0,0]
    elif point=='outside':
        # T_o
        sigmaEqMax = np.interp(s.T[0,-1], f[:,0], f[:,i])
        return sigmaEqMax - s.sigmaEq[0,-1]
    elif point=='membrane':
        # Assuming shakedown has occured (membrane stress remains):
        sigmaEqMax = np.interp(np.average(s.T[0,:]), f[:,0], f[:,i])
        return sigmaEqMax - np.average(s.sigmaEq[0,:])
    else: sys.exit('Variable point {} not recognised'.format(point))

################################### ROUTINES ###################################

def Timoshenko1951():
    """
    Timoshenko, S. and Goodier, J. N., 1951. Theory of Elasticity.
    McGraw-Hill Book Company, Inc., 2nd edn. 1st ed. 1934.
    """
    headerprint(' Case 135 p412 Timoshenko & Goodier, 1951 ')

    nr=10; nt=32
    a = 500e-3      # inside tube radius [mm->m]
    b = 700e-3      # outside tube radius [mm->m]

    E = 200e9
    valprint('E', E*1e-9, 'GPa')
    alpha = 1e-5
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')

    # Create instance of Grid and Solver to use stress post-processor:
    GPS = True # True: generalised plane strain (default); False: simple plane strain.
    g = Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = Solver(g, debug=True, alpha=alpha,
               E=E, nu=0.3, n=1, GPS=GPS)
    s.T = 100 * ((np.log(g.r / a) / np.log(b / a)))
    s.postProcessing()

    # plotStress(g.theta, g.r, s.sigmaR,
    #            s.sigmaR.min(), s.sigmaR.max(),
    #            'Timoshenko_sigmaR.pdf')
    # plotStress(g.theta, g.r, s.sigmaTheta,
    #            s.sigmaTheta.min(), s.sigmaTheta.max(),
    #            'Timoshenko_sigmaTheta.pdf')
    # plotStress(g.theta, g.r, s.sigmaRTheta,
    #            s.sigmaRTheta.min(), s.sigmaRTheta.max(),
    #            'Timoshenko_sigmaRTheta.pdf')
    # plotStress(g.theta, g.r, s.sigmaZ,
    #            s.sigmaZ.min(), s.sigmaZ.max(),
    #            'Timoshenko_sigmaZ.pdf')
    # plotStress(g.theta, g.r, s.sigmaEq,
    #            s.sigmaEq.min(), s.sigmaEq.max(),
    #            'Timoshenko_sigmaEq.pdf')

    """ Comparison with code_aster 13.6 MECA_STATIQUE [U4.51.01]: """
    strain = 'GPS' if GPS else 'SPS'
    feaCmp = np.genfromtxt(
        os.path.join('aster', 'TIMOSHENKO_{}'.format(strain)+\
                     '_THETA0_SIEF-CYL.dat'),
        skip_header=5
    )
    feaEq = np.genfromtxt(
        os.path.join('aster', 'TIMOSHENKO_{}'.format(strain)+\
                     '_THETA0_SIEQ.dat'),
        skip_header=5
    )
    plotTIMO(g.r, s, feaCmp, feaEq, 'Timoshenko-{}.pdf'.format(strain))

def Holms1952():
    """
    Holms, A.G., 1952. A biharmonic relaxation method for calculating thermal
    stress in cooled irregular cylinders. Technical Report NACA-TR-1059.
    Lewis Flight Propulsion Lab. Cleveland.
    URL: https://ntrs.nasa.gov/search.jsp?R=19930092105
    """
    headerprint(' Holms (1952), NACA-TR-1059 ')

    nr=34; nt=61
    a = 101.6/1e3      # inside tube radius [mm->m]
    b = 304.8/1e3      # outside tube radius [mm->m]

    c0 = Q_(0,'degF').to('K').magnitude
    c1 = Q_(1000,'degF').to('K').magnitude
    c2 = Q_(500,'degF').to('K').magnitude
    E = Q_(17.5e6, 'psi').to('Pa').magnitude
    valprint('E', E*1e-9, 'GPa')
    alpha = Q_(8e-6, 'degF^-1').to('K^-1').magnitude
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')

    # Create instance of Grid and Solver to use stress post-processor:
    g = Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = Solver(g, debug=True, alpha=alpha,
               E=E, nu=0.3, n=1)
    s.T = (((c1-c0) * b) / (b**2 - a**2)) * \
          ((g.r**2 - a**2) / g.r) * np.cos(g.theta) + \
          (c2-c0) * (1 - (np.log(b / g.r) / np.log(b / a)))
    s.postProcessing()

    # plotStress(g.theta, g.r, s.sigmaR,
    #            s.sigmaR.min(), s.sigmaR.max(),
    #            'NACA-TR-1059_sigmaR.pdf')
    # plotStress(g.theta, g.r, s.sigmaTheta,
    #            s.sigmaTheta.min(), s.sigmaTheta.max(),
    #            'NACA-TR-1059_sigmaTheta.pdf')
    # plotStress(g.theta, g.r, s.sigmaRTheta,
    #            s.sigmaRTheta.min(), s.sigmaRTheta.max(),
    #            'NACA-TR-1059_sigmaRTheta.pdf')
    # plotStress(g.theta, g.r, s.sigmaZ,
    #            s.sigmaZ.min(), s.sigmaZ.max(),
    #            'NACA-TR-1059_sigmaZ.pdf')
    # plotStress(g.theta, g.r, s.sigmaEq,
    #            s.sigmaEq.min(), s.sigmaEq.max(),
    #            'NACA-TR-1059_sigmaEq.pdf')

    headerprint('Table II, p89: radius vs. tangential (hoop) stress', ' ')
    radius_inch = np.linspace(4,12,9)
    radius = Q_(radius_inch, 'inch').to('m')
    sigmaTheta = Q_(np.interp(radius, s.g.r[0,:], s.sigmaTheta[0,:]), 'Pa').to('psi')
    for r, sig_t in zip(radius_inch, sigmaTheta):
        valprint('{} (in.)'.format(r), sig_t.magnitude, 'psi')

    """ Comparison with code_aster 13.6 MECA_STATIQUE [U4.51.01]: """
    fea = [None]*4
    for i, theta in enumerate([0, 60, 120, 180]):
        fn = 'NACA-TR-1059_THETA{}'.format(theta) + \
             '_SIEF-CYL.dat'
        fea[i] = np.genfromtxt(os.path.join('aster', fn), skip_header=5)
    plotNACA(g.r, s.sigmaTheta, fea, 4, 'NACA-TR-1059_sigmaTheta.pdf', \
             'best', r'\textsc{hoop stress}, $\sigma_\theta$')
    plotNACA(g.r, s.sigmaR, fea, 5, 'NACA-TR-1059_sigmaR.pdf', \
             'best', r'\textsc{radial stress}, $\sigma_r$')
    plotNACA(g.r, s.sigmaZ, fea, 6, 'NACA-TR-1059_sigmaZ.pdf', \
             'best', r'\textsc{axial stress}, $\sigma_z$')
    plotNACA(g.r, s.sigmaRTheta, fea, 7, 'NACA-TR-1059_sigmaRTheta.pdf', \
             'best', r'\textsc{in-plane shear stress}, $\tau_{r\theta}$')
    fea = [None]*4
    for i, theta in enumerate([0, 60, 120, 180]):
        fn = 'NACA-TR-1059_THETA{}'.format(theta) + \
             '_SIEQ.dat'
        fea[i] = np.genfromtxt(os.path.join('aster', fn), skip_header=5)
    plotNACA(g.r, s.sigmaEq, fea, 1, 'NACA-TR-1059_sigmaEq.pdf', \
             'best', r'\textsc{equivalent stress}, $\sigma_\mathrm{eq}$')

def SE6413():
    """
    Reproduction of results from Solar Energy 160 (2018) 368-379
    https://doi.org/10.1016/j.solener.2017.12.003
    """

    headerprint(' NPS Sch. 5S 1" S31609 at 450degC ')

    nr=12; nt=91
    a = 30.098/2e3     # inside tube radius [mm->m]
    b = 33.4/2e3       # outside tube radius [mm->m]

    """ Create instance of Grid: """
    g = Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution

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
    s.extBC = s.tubeExtCosFluxRadConv
    #s.extBC = s.tubeExtCosFluxRadConvAdiabaticBack

    """ Internal BC: """
    #s.intBC = s.tubeIntTemp
    #s.intBC = s.tubeIntFlux
    s.intBC = s.tubeIntConv

    salt = nitrateSalt(True); salt.update(723.15)
    h_int, dP = HTC(True, salt, a, b, 20, 'Dittus', 'mdot', 5)

    headerprint(' HTC: {} W/m^s/K '.format(h_int), ' ')
    s.h_int = h_int
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

    plotTemperatureAnnotate(g.theta, g.r, s.T,
                            s.T.min(), s.T.max(),
                            'S31609_nitrateSalt_T.pdf')
    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(),
               'S31609_nitrateSalt_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(),
               'S31609_nitrateSalt_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta,
               s.sigmaRTheta.min(), s.sigmaRTheta.max(),
               'S31609_nitrateSalt_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ,
               s.sigmaZ.min(), s.sigmaZ.max(),
               'S31609_nitrateSalt_sigmaZ.pdf')
    plotStressAnnotate(g.theta, g.r, s.sigmaEq,
                       s.sigmaEq.min(), s.sigmaEq.max(),
                       'right', 'S31609_nitrateSalt_sigmaEq.pdf')

    sodium = liquidSodium(True); sodium.update(723.15)
    h_int, dP = HTC(True, sodium, a, b, 20, 'Skupinski', 'mdot', 4)

    headerprint(' HTC: {} W/m^s/K '.format(h_int), ' ')
    s.h_int = h_int
    t = time.clock(); ret = s.solve(eps=1e-6); s.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    plotTemperatureAnnotate(g.theta, g.r, s.T,
                            s.T.min(), s.T.max(),
                            'S31609_liquidSodium_T.pdf')
    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(),
               'S31609_liquidSodium_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(),
               'S31609_liquidSodium_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta,
               s.sigmaRTheta.min(), s.sigmaRTheta.max(),
               'S31609_liquidSodium_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ,
               s.sigmaZ.min(), s.sigmaZ.max(),
               'S31609_liquidSodium_sigmaZ.pdf')
    plotStressAnnotate(g.theta, g.r, s.sigmaEq,
                       s.sigmaEq.min(), s.sigmaEq.max(),
                       'right', 'S31609_liquidSodium_sigmaEq.pdf')

def ASTRI2():
    """
    Peak flux reference point in advanced CSP prototypes using:
    -- N06625 tube (25.4mm OD x 1.65mm WT)
    -- N06230 tube (33.4mm OD x 1.32mm WT)
    """
    headerprint(' 25.4mm OD x 1.65mm WT N06625 at 650 degC ')

    ## Material constants
    b = 25.4e-3/2.     # inside tube radius (mm->m)
    valprint('b', b*1e3, 'mm')
    a = b - 1.65e-3    # outside tube radius (mm->m)
    valprint('a', a*1e3, 'mm')
    k = 19.1           # thermal conductivity (kg*m/s^3/K)
    valprint('k', k, 'kg*m/s^3/K')
    alpha = 18.2e-6  # thermal dilation (K^-1)
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')
    E = 169e9          # Youngs modulus (Pa)
    valprint('E', E*1e-9, 'GPa')
    valprint('TSP', 1e-3*(alpha*E)/k, 'kPa.m/W')
    nu = 0.31          # Poisson
    valprint('nu', nu)
    rho = 8440         # density kg/m^3
    m_t = (pi*b**2-pi*a**2)*rho

    ## Thermal constants
    CG = 7.65e5         # absorbed flux (W/m^2)
    valprint('CG', CG*1e-3, 'kW/m^2')
    mdot = 0.1         # mass flow (kg/s)
    valprint('mdot', mdot, 'kg/s')
    T_int = 888        # bulk sodium temperature (K)
    sodium = liquidSodium(True); sodium.update(T_int)
    #h_int, dP = HTC(True, sodium, a, b, k, 'Skupinski', 'velocity', 4.0)
    #h_int, dP = HTC(True, sodium, a, b, k, 'Skupinski', 'heatCapRate', 5000)
    #h_int, dP = HTC(True, sodium, a, b, k, 'Skupinski', 'mdot', mdot)
    #h_int, dP = HTC(True, sodium, a, b, k, 'Notter', 'mdot', mdot)
    h_int, dP = HTC(True, sodium, a, b, k, 'Chen', 'mdot', mdot)
    m_s = pi*a**2*sodium.rho

    ## Mechanical constants
    valprint('dead-weight', m_t+m_s, 'kg/m') # dead weight of tube + fluid
    P_i = 6e5 # Pa
    valprint('P_i', P_i*1e-5, 'bar (x1e-5 Pa)')

    """ Create instance of Grid: """
    nr = 17; nt = 61
    gN06625 = Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution

    """ Create instance of LaplaceSolver: """
    sN06625 = Solver(gN06625, debug=True, CG=CG, k=k, T_int=T_int, R_f=0,
               h_int=h_int, P_i=P_i, alpha=alpha, E=E, nu=nu, n=1,
               bend=False)

    ## Generalised plane strain:
    headerprint('Generalised plane strain (pressure only)', ' ')
    ## If post-processing is called before s.solve thermal field is at T_int:
    sN06625.postProcessing()

    """ Comparison with code_aster 13.6 MECA_STATIQUE [U4.51.01]: """
    fea = [None]*4
    for i, theta in enumerate([0, 60, 120, 180]):
        fn = 'N06625_GPS_THETA{}'.format(theta) + \
             '_PR600_SIEF-CYL.dat'
        fc = np.genfromtxt(os.path.join('aster', fn), skip_header=5)
        seq = np.sqrt(0.5 * ((fc[:,5] - fc[:,4])**2 + \
                            (fc[:,4] - fc[:,6])**2 + \
                            (fc[:,6] - fc[:,5])**2) + \
                     6 * (fc[:,7]**2))
        fea[i] = np.c_[fc, seq]
    plotASTER(gN06625.r, sN06625.sigmaTheta, fea, 4, 'N06625_GPS_sigmaTheta_PR_FEA.pdf',
            'best', r'\textsc{hoop stress}, $\sigma_\theta$')
    plotASTER(gN06625.r, sN06625.sigmaR, fea, 5, 'N06625_GPS_sigmaR_PR_FEA.pdf',
            'best', r'\textsc{radial stress}, $\sigma_r$')
    plotASTER(gN06625.r, sN06625.sigmaZ, fea, 6, 'N06625_GPS_sigmaZ_PR_FEA.pdf',
            'best', r'\textsc{axial stress}, $\sigma_z$')
    plotASTER(gN06625.r, sN06625.sigmaRTheta, fea, 7, 'N06625_GPS_sigmaRTheta_PR_FEA.pdf',
            'best', r'\textsc{in-plane shear stress}, $\sigma_{r\theta}$')
    plotASTER(gN06625.r, sN06625.sigmaEq, fea, 8, 'N06625_GPS_sigmaEq_PR_FEA.pdf',
            'best', r'\textsc{equiv. stress}, $\sigma_{\mathrm{Eq}}$')

    headerprint('Generalised plane strain (thermal only)', ' ')
    sN06625.P_i = 0e5

    """ External BC: """
    #sN06625.extBC = sN06625.tubeExtTemp
    sN06625.extBC = sN06625.tubeExtCosFlux
    #sN06625.extBC = sN06625.tubeExtConv
    #sN06625.extBC = sN06625.tubeExtCosFluxRadConv
    #sN06625.extBC = sN06625.tubeExtCosFluxRadConvAdiabaticBack

    """ Internal BC: """
    #sN06625.intBC = sN06625.tubeIntTemp
    #sN06625.intBC = sN06625.tubeIntFlux
    sN06625.intBC = sN06625.tubeIntConv

    """ Run LaplaceSolver """
    t = time.clock(); ret = sN06625.solve(eps=1e-6)
    sN06625.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaR,
               sN06625.sigmaR.min(), sN06625.sigmaR.max(),
               'N06625_GPS_sigmaR_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaTheta,
               sN06625.sigmaTheta.min(), sN06625.sigmaTheta.max(),
               'N06625_GPS_sigmaTheta_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaRTheta,
               sN06625.sigmaRTheta.min(), sN06625.sigmaRTheta.max(),
               'N06625_GPS_sigmaRTheta_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaZ,
               sN06625.sigmaZ.min(), sN06625.sigmaZ.max(),
               'N06625_GPS_sigmaZ_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaEq,
               sN06625.sigmaEq.min(), sN06625.sigmaEq.max(),
               'N06625_GPS_sigmaEq_TH.pdf')

    """ Comparison with code_aster 13.6 MECA_STATIQUE [U4.51.01]: """
    fea = [None]*4
    for i, theta in enumerate([0, 60, 120, 180]):
        fn = 'N06625_GPS_THETA{}'.format(theta) + \
             '_TSOD615_HTCSOD17394_FLUX765' + \
             '_SIEF-CYL.dat'
        fc = np.genfromtxt(os.path.join('aster', fn), skip_header=5)
        seq = np.sqrt(0.5 * ((fc[:,5] - fc[:,4])**2 + \
                            (fc[:,4] - fc[:,6])**2 + \
                            (fc[:,6] - fc[:,5])**2) + \
                     6 * (fc[:,7]**2))
        fea[i] = np.c_[fc, seq]
    plotASTER(gN06625.r, sN06625.sigmaTheta, fea, 4, 'N06625_GPS_sigmaTheta_TH_FEA.pdf',
            'best', r'\textsc{hoop stress}, $\sigma_\theta$')
    plotASTER(gN06625.r, sN06625.sigmaR, fea, 5, 'N06625_GPS_sigmaR_TH_FEA.pdf',
            'best', r'\textsc{radial stress}, $\sigma_r$')
    plotASTER(gN06625.r, sN06625.sigmaZ, fea, 6, 'N06625_GPS_sigmaZ_TH_FEA.pdf',
            'best', r'\textsc{axial stress}, $\sigma_z$')
    plotASTER(gN06625.r, sN06625.sigmaRTheta, fea, 7, 'N06625_GPS_sigmaRTheta_TH_FEA.pdf',
            'best', r'\textsc{in-plane shear stress}, $\sigma_{r\theta}$')
    plotASTER(gN06625.r, sN06625.sigmaEq, fea, 8, 'N06625_GPS_sigmaEq_TH_FEA.pdf',
            'best', r'\textsc{equiv. stress}, $\sigma_{\mathrm{Eq}}$')

    ## Generalised plane strain with annulled bending:
    sN06625.bend = True
    headerprint('Generalised plane strain with annulled bending moment', ' ')
    sN06625.postProcessing()

    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaR,
               sN06625.sigmaR.min(), sN06625.sigmaR.max(),
               'N06625_GPS-AB_sigmaR_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaTheta,
               sN06625.sigmaTheta.min(), sN06625.sigmaTheta.max(),
               'N06625_GPS-AB_sigmaTheta_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaRTheta,
               sN06625.sigmaRTheta.min(), sN06625.sigmaRTheta.max(),
               'N06625_GPS-AB_sigmaRTheta_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaZ,
               sN06625.sigmaZ.min(), sN06625.sigmaZ.max(),
               'N06625_GPS-AB_sigmaZ_TH.pdf')
    plotStress(gN06625.theta, gN06625.r, sN06625.sigmaEq,
               sN06625.sigmaEq.min(), sN06625.sigmaEq.max(),
               'N06625_GPS-AB_sigmaEq_TH.pdf')

    headerprint('Determining peak flux for N06625', ' ')
    mdot = 0.2         # mass flow (kg/s)
    sN06625.debug = False; sN06625.bend = False
    sodium.debug = False
    fv = np.genfromtxt(os.path.join('mats', 'N06625_f-values.dat'), delimiter=',')
    fv[:,0] += 273.15 # degC to K
    fv[:,1:] *= 3e6 # apply 3f criteria and convert MPa->Pa
    #nfv = 5 # f in 1e2, 1e3, 1e4 and 1e5 hours, as well as ASME S_m
    nfv = 4 # f in 1e2, 1e3 and 1e4 hours, as well as ASME S_m
    T_int = np.linspace(500, 750, 11)+273.15
    T_met = np.zeros([len(T_int), nfv])
    peakFlux = np.zeros([len(T_int), nfv])
    t = time.clock()
    for i in range(len(T_int)):
        sN06625.T_int = T_int[i]
        sodium.update(T_int[i])
        sN06625.h_int, dP = HTC(False, sodium, a, b, k, 'Chen', 'mdot', mdot)
        for j in range(nfv):
            peakFlux[i, j] = opt.newton(
                findFlux, 1e5,
                args=(sN06625, fv, j+1, 'outside'),
                maxiter=1000, tol=1e-2
            )
            T_met[i, j] = np.max(sN06625.T)
    valprint('Time taken', time.clock() - t, 'sec')

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(T_int-273.15,peakFlux[:,0]*1e-6, label=r'$f$ \textsc{in} \SI{100}{\hour}')
    ax.plot(T_int-273.15,peakFlux[:,1]*1e-6, label=r'$f$ \textsc{in} \SI{1000}{\hour}')
    ax.plot(T_int-273.15,peakFlux[:,2]*1e-6, label=r'$f$ \textsc{in} \SI{10000}{\hour}')
    #ax.plot(T_int-273.15,peakFlux[:,3]*1e-6, label=r'$f$ \textsc{in} \SI{100000}{\hour}')
    ax.plot(T_int-273.15,peakFlux[:,3]*1e-6, label=r'$f=S_\mathrm{m}$')
    ax.set_xlabel(r'\textsc{sodium temperature}, $T_\mathrm{bulk}$ (\si{\celsius})')
    ax.set_ylabel(r'\textsc{peak flux}, (\si{\mega\watt\per\meter\squared})')
    ax.set_ylim(0.2, 1.6)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig('N06625_OD25-4_WT1-65_peakFlux.pdf')
    fig.savefig('N06625_OD25-4_WT1-65_peakFlux.png', dpi=150)
    plt.close(fig)
    ## Dump peak flux results to CSV file:
    csv = np.c_[T_int,
                T_met[:,0], peakFlux[:,0],
                T_met[:,1], peakFlux[:,1],
                T_met[:,2], peakFlux[:,2],
                #T_met[:,3], peakFlux[:,3],
                T_met[:,3], peakFlux[:,3]
    ]
    np.savetxt('N06625_OD25-4_WT1-65_peakFlux.csv', csv, delimiter=',',
               header='T_int(K),'+\
               'T_metal@3f_in_1e2h(K),flux_in_1e2h(W/(m^2.K)),'+\
               'T_metal@3f_in_1e3h(K),flux_in_1e3h(W/(m^2.K)),'+\
               'T_metal@3f_in_1e4h(K),flux_in_1e4h(W/(m^2.K)),'+\
               #'T_metal@3f_in_1e5h(K),flux_in_1e5h(W/(m^2.K)),'+\
               'T_metal@3Sm(K),flux_in_Sm(W/(m^2.K))'
    )

    headerprint(' 33.4mm OD x 1.32mm WT N06320 at 650 degC ')

    ## Material constants
    b = 33.4e-3/2.     # inside tube radius (mm->m)
    valprint('b', b*1e3, 'mm')
    a = b - 1.32e-3    # outside tube radius (mm->m)
    valprint('a', a*1e3, 'mm')
    k = 21.4           # thermal conductivity (kg*m/s^3/K)
    valprint('k', k, 'kg*m/s^3/K')
    alpha = 16.5e-6  # thermal dilation (K^-1)
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')
    E = 172e9          # Youngs modulus (Pa)
    valprint('E', E*1e-9, 'GPa')
    valprint('TSP', 1e-3*(alpha*E)/k, 'kPa.m/W')
    nu = 0.31          # Poisson
    valprint('nu', nu)
    rho = 8220 # density kg/m^3
    m_t = (pi*b**2-pi*a**2)*rho

    ## Thermal constants
    CG = 7.65e5         # absorbed flux (W/m^2)
    valprint('CG', CG*1e-3, 'kW/m^2')
    mdot = 0.1         # mass flow (kg/s)
    valprint('mdot', mdot, 'kg/s')
    T_int = 888        # bulk sodium temperature (K)
    sodium = liquidSodium(True); sodium.update(T_int)
    h_int, dP = HTC(True, sodium, a, b, k, 'Chen', 'mdot', mdot)
    m_s = pi*a**2*sodium.rho

    ## Mechanical constants
    valprint('dead-weight', m_t+m_s, 'kg/m') # dead weight of tube + fluid
    P_i = 0e5 # Pa
    valprint('P_i', P_i*1e-5, 'bar (x1e-5 Pa)')

    """ Create instance of Grid: """
    nr = 17; nt = 61
    gN06230 = Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution

    """ Create instance of LaplaceSolver: """
    sN06230 = Solver(gN06230, debug=False, CG=CG, k=k, T_int=T_int, R_f=0,
               h_int=h_int, P_i=P_i, alpha=alpha, E=E, nu=nu, n=1,
               bend=False)

    """ External BC: """
    sN06230.extBC = sN06230.tubeExtCosFlux

    """ Internal BC: """
    sN06230.intBC = sN06230.tubeIntConv

    headerprint('Determining peak flux for N06230', ' ')
    mdot = 0.2         # mass flow (kg/s)
    sN06230.debug = False
    sodium.debug = False
    fv = np.genfromtxt(os.path.join('mats', 'N06230_f-values.dat'), delimiter=',')
    fv[:,0] += 273.15 # degC to K
    fv[:,1:] *= 3e6 # apply 3f criteria and convert MPa->Pa
    #nfv = 5 # f in 1e2, 1e3, 1e4 and 1e5 hours, as well as ASME S_m
    nfv = 4 # f in 1e2, 1e3 and 1e4 hours, as well as ASME S_m
    T_int = np.linspace(500, 750, 11)+273.15
    T_met = np.zeros([len(T_int), nfv])
    peakFlux = np.zeros([len(T_int), nfv])
    t = time.clock()
    for i in range(len(T_int)):
        sN06230.T_int = T_int[i]
        sodium.update(T_int[i])
        sN06230.h_int, dP = HTC(False, sodium, a, b, k, 'Chen', 'mdot', mdot)
        for j in range(nfv):
            peakFlux[i, j] = opt.newton(
                findFlux, 1e5,
                args=(sN06230, fv, j+1, 'outside'),
                maxiter=1000, tol=1e-2
            )
            T_met[i, j] = np.max(sN06230.T)
    valprint('Time taken', time.clock() - t, 'sec')

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(T_int-273.15,peakFlux[:,0]*1e-6, label=r'$f$ \textsc{in} \SI{100}{\hour}')
    ax.plot(T_int-273.15,peakFlux[:,1]*1e-6, label=r'$f$ \textsc{in} \SI{1000}{\hour}')
    ax.plot(T_int-273.15,peakFlux[:,2]*1e-6, label=r'$f$ \textsc{in} \SI{10000}{\hour}')
    ax.plot(T_int-273.15,peakFlux[:,3]*1e-6, label=r'$f=S_\mathrm{m}$')
    ax.set_xlabel(r'\textsc{sodium temperature}, $T_\mathrm{bulk}$ (\si{\celsius})')
    ax.set_ylabel(r'\textsc{peak flux}, (\si{\mega\watt\per\meter\squared})')
    ax.set_ylim(0.2, 1.6)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig('N06230_OD33-4_WT1-32_peakFlux.pdf')
    fig.savefig('N06230_OD33-4_WT1-32_peakFlux.png', dpi=150)
    plt.close(fig)
    ## Dump peak flux results to CSV file:
    csv = np.c_[T_int,
                T_met[:,0], peakFlux[:,0],
                T_met[:,1], peakFlux[:,1],
                T_met[:,2], peakFlux[:,2],
                T_met[:,3], peakFlux[:,3]
    ]
    np.savetxt('N06230_OD33-4_WT1-32_peakFlux.csv', csv, delimiter=',',
               header='T_int(K),'+\
               'T_metal@3f_in_1e2h(K),flux_in_1e2h(W/(m^2.K)),'+\
               'T_metal@3f_in_1e3h(K),flux_in_1e3h(W/(m^2.K)),'+\
               'T_metal@3f_in_1e4h(K),flux_in_1e4h(W/(m^2.K)),'+\
               'T_metal@3Sm(K),flux_in_Sm(W/(m^2.K))'
    )

    #headerprint(' COMPARISON OF N06625 AND N06230 ')

    ## Sensitivity of heat transfer coefficient (pressure drop) and stress to mass-flow
    T_int = 888; sodium.update(T_int)
    sN06625.T_int = T_int; sN06625.CG = CG
    sN06230.T_int = T_int; sN06230.CG = CG
    mdot = np.linspace(0.05, 1, 51)
    h_N06625 = np.zeros(len(mdot))
    sig_N06625 = np.zeros(len(mdot))
    dP_N06625 = np.zeros(len(mdot))
    h_N06230 = np.zeros(len(mdot))
    sig_N06230 = np.zeros(len(mdot))
    dP_N06230 = np.zeros(len(mdot))
    for i, m in enumerate(mdot):
        h_N06625[i], dP_N06625[i] = HTC(
            False, sodium, gN06625.a, gN06625.b,
            sN06625.k, 'Chen', 'mdot', m
        )
        sN06625.h_int = h_N06625[i]
        ret = sN06625.solve(eps=1e-6)
        sN06625.postProcessing()
        sig_N06625[i] = sN06625.sigmaEq[0,-1]
        h_N06230[i], dP_N06230[i] = HTC(
            False, sodium, gN06230.a, gN06230.b,
            sN06230.k, 'Chen', 'mdot', m
        )
        sN06230.h_int = h_N06230[i]
        ret = sN06230.solve(eps=1e-6)
        sN06230.postProcessing()
        sig_N06230[i] = sN06230.sigmaEq[0,-1]
    fig = plt.figure(figsize=(4, 3.5))
    ax1 = fig.add_subplot(111)
    c1 = 'C2'; c2 = 'C3'
    N06625_m = Line2D([], [], marker='o', color='k', label='N06625')
    N06230_m = Line2D([], [], marker='x', color='k', label='N06230')
    ax1.plot(mdot, h_N06625*1e-3, 'o-', color=c1, markevery=5)
    ax1.plot(mdot, h_N06230*1e-3, 'x-', color=c1, markevery=5)
    ax1.set_xlabel(r'\textsc{mass flow}, $\dot{m}$ (\si{\kilo\gram\per\second})')
    ax1.set_ylabel(r'\textsc{int. heat transfer}, $h_\mathrm{int}$ ' + \
                   '(\si{\kilo\watt\per\meter\squared\per\kelvin})', color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    ax2 = ax1.twinx()
    ax2.plot(mdot, -dP_N06625*1e-3, 'o-', color=c2, markevery=5)
    ax2.plot(mdot, -dP_N06230*1e-3, 'x-', color=c2, markevery=5)
    ax2.set_ylabel(r'\textsc{pressure drop}, $\Delta P$' + \
                   ' (\si{\kilo\pascal\per\meter})', color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    ax1.legend(loc='best')
    ax1.legend(loc='best', handles=[N06625_m, N06230_m])
    fig.tight_layout()
    fig.savefig('N06625vN06230_mdot-intConv.pdf')
    #fig.savefig('N06625vN06230_mdot-intConv.png', dpi=150)
    plt.close(fig)
    ## plot of mdot vs maximum equivalent stress
    fig = plt.figure(figsize=(3.5, 3.5))
    ax3 = fig.add_subplot(111)
    ax3.plot(mdot, sig_N06625*1e-6, 'o-', label='N06625', markevery=5)
    ax3.plot(mdot, sig_N06230*1e-6, 'x-', label='N06230', markevery=5)
    ax3.set_xlabel(r'\textsc{mass flow}, $\dot{m}$ (\si{\kilo\gram\per\second})')
    ax3.set_ylabel(r'\textsc{max. equivalent stress}, $\max\sigma_\mathrm{Eq}$ (MPa)')
    ax3.legend(loc='best')
    fig.tight_layout()
    fig.savefig('N06625vN06230_mdot-sigmaEq.pdf')
    #fig.savefig('N06625vN06230_mdot.pdf')
    #fig.savefig('N06625vN06230_mdot-sigmaEq.png', dpi=150)
    plt.close(fig)

##################################### MAIN #####################################

if __name__ == "__main__":

    # Timoshenko1951()
    # Holms1952()
    # SE6413()
    # ASTRI2()
    pass
