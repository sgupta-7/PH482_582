'''
Using given cosmology and transfer function, calculate various analytic parameters using linear
density field vs extended press-schechter formalism.

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [log M_sun/h]
    distance [kpc/h comoving]
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
from scipy import integrate, interpolate
# local ----
from . import basic as ut
from . import cosmology

# spline ranges
REDSHIFT_LIMITS = [0.0, 10.0]  # redshift range
REDSHIFT_BIN_NUMBER = 100
MASS_LIMITS = [5.0, 16.0]  # min and max log masses
MASS_BIN_NUMBER = 300

# cosmology parameters
COSMOLOGY = cosmology.CosmologyClass(source='planck')
DELTA_C = 1.686  # threshold in linear over-density for halo collapse


#===================================================================================================
# growth function
#===================================================================================================
class GrowthClass:
    '''
    Get value and derivative of cosmic growth factor as a function of redshift.
    '''

    def __init__(self, redshift_limits=REDSHIFT_LIMITS, redshift_bin_number=REDSHIFT_BIN_NUMBER,
                 Cosmology=COSMOLOGY):
        '''
        Fit and store growth(z) to spline in redshift range.

        Parameters
        ----------
        redshift_limits : list : min and max limits of redshift
        redshift_bin_number : int : number of spline points within limits
        Cosmology : class : cosmology class
        '''
        self.Cosmology = Cosmology
        self.redshifts = np.linspace(redshift_limits[0], redshift_limits[1], redshift_bin_number)
        self.growths = np.zeros(redshift_bin_number)
        for zi in range(self.redshifts.size):
            self.growths[zi] = self.Cosmology.get_growth_factor(self.redshifts[zi])

        self.growth_redshift_spl = interpolate.splrep(self.redshifts, self.growths)

    def value(self, redshifts):
        '''
        Parameters
        ----------
        redshifts : float or array
        '''
        return interpolate.splev(redshifts, self.growth_redshift_spl)

    def derivative(self, redshifts):
        '''
        Parameters
        ----------
        redshifts : float or array
        '''
        return interpolate.splev(redshifts, self.growth_redshift_spl, der=1)


#===================================================================================================
# mass variance
#===================================================================================================
class MassVarianceClass:
    '''
    Get value and derivative of log variance of mass as a function of log mass and redshift.
    '''

    def __init__(self, Growth=None, mass_limits=MASS_LIMITS, mass_bin_number=MASS_BIN_NUMBER,
                 redshift_limits=REDSHIFT_LIMITS, redshift_bin_number=REDSHIFT_BIN_NUMBER,
                 log_k_limits=[-15, 15], Cosmology=None):
        '''
        Calculate log sigma(mass) in mass range, normalized to Cosmology['sigma_8'],
        store as spline.
        Valid at z = 0, use growth function to scale to other redshifts.

        Parameters
        ----------
        Growth : class : growth function class
        mass_limits : list : min and max limits of mass
        mass_bin_number : int : number of spline points within limits
        redshift_limits : list : min and max limits of redshift
        redshift_bin_number : int : number of spline points within limits
        log_k_limits : list : min and max limits of wavenumber
            use for integral over P(k) to get sigma(mass) [h/kpc]
        Cosmology : class : cosmology class
        '''
        if Cosmology is not None:
            self.Cosmology = Cosmology
        else:
            if Growth is not None:
                self.Cosmology = Growth.Cosmology
            else:
                self.Cosmology = COSMOLOGY

        if Growth is not None:
            self.Growth = Growth
        else:
            self.Growth = GrowthClass(
                redshift_limits, redshift_bin_number, Cosmology=self.Cosmology)

        self.log_k_limits = log_k_limits
        self.masses = np.linspace(mass_limits[0], mass_limits[1], mass_bin_number)
        self.log_sigmas = np.zeros(mass_bin_number)

        for mi in range(self.masses.size):
            self.log_sigmas[mi] = self.log_sigma(self.masses[mi])

        # normalize so sigma(M(8000 kpc/h)) = self.Cosmology.sigma8
        # mass density [M_sun / (kpc/h)^3 comoving] at z = 0
        mass_density = self.Cosmology.get_density('matter', 0, 'kpc/h comoving')
        log_m8 = np.log10(mass_density * 8000 ** 3)  # mass within 8000 kpc/h at z = 0
        log_sigma8 = self.log_sigma(log_m8)  # sigma for above mass
        dlog_sigma = np.log10(self.Cosmology['sigma_8']) - log_sigma8
        self.log_sigmas += dlog_sigma
        self.log_sigmas_ms_spl = interpolate.splrep(self.masses, self.log_sigmas)

        # set Delta_h
        #Delta_h = self.Cosmology['sigma_8'] / 10 ** log_sigma8

    def log_sigma(self, mass):
        '''
        Get log sigma(mass) for mass at z = 0.
        Note: sigma is *not* normalized to Cosmology['sigma_8'].

        Parameters
        ----------
        mass : float : [log M_sun/h]
        '''

        def get_dsigma(log_k, radius, delta2_func):
            wk = get_window_function(10 ** log_k * radius)
            return delta2_func(10 ** log_k) * wk ** 2

        def get_window_function(kr):
            if kr < 0.2:
                return 1 + kr ** 2 * (-0.1 + kr ** 2 * (1 / 280 - kr ** 2 / 15120))
            else:
                return 3 * (np.sin(kr) - kr * np.cos(kr)) / kr ** 3

        # mass density [M_sun / (kpc/h)^3 comoving] at z = 0
        density = self.Cosmology.get_density('matter', 0, 'kpc/h comoving')
        radius = (10 ** mass / density) ** (1 / 3)  # linear radius for given mass
        sigma = integrate.quad(get_dsigma, self.log_k_limits[0], self.log_k_limits[1],
                               (radius, self.Cosmology.delta2), epsrel=2e-3)[0]

        return 0.5 * np.log10(sigma)

    def value(self, mass, redshifts=0):
        '''
        Parameters
        ----------
        mass : float : [log M_sun/h]
        redshifts : float or array
        '''
        # sigma at z = 0 (no growth function)
        log_sigma = interpolate.splev(mass, self.log_sigmas_ms_spl)
        # scale by growth function to get to sigma(z)
        if redshifts > 0:
            log_sigma += np.log10(self.Growth.value(redshifts))

        return log_sigma

    def derivative(self, mass):
        '''
        Parameters
        ----------
        mass : float : [log M_sun/h]

        Note: not need to scale by growth function because this is derivative in log.
        '''

        return interpolate.splev(mass, self.log_sigmas_ms_spl, der=1)


class MassClass:
    '''
    Get value and derivative of log mass as a function of log variance of mass.
    '''

    def __init__(self, Sigma=None, mass_limits=MASS_LIMITS, mass_bin_number=MASS_BIN_NUMBER,
                 redshift_limits=REDSHIFT_LIMITS, redshift_bin_number=REDSHIFT_BIN_NUMBER,
                 Growth=None, Cosmology=None):
        '''
        Make spline of log mass v log variance of mass.

        Parameters
        ----------
        Sigma : class : sigma(mass) class
        mass_limits : list : min and max limits of mass
        mass_bin_number : int : number of spline points within limits
        redshift_limits : list : min and max limits of redshift
        redshift_bin_number : int : number of spline points within limits
        Growth : class : growth function class
        Cosmology : class : cosmology class
        '''
        if Cosmology is not None:
            self.Cosmology = Cosmology
        else:
            if Growth is not None:
                self.Cosmology = Growth.Cosmology
            elif Sigma is not None:
                self.Cosmology = Sigma.Cosmology
            else:
                self.Cosmology = COSMOLOGY

        if Growth is not None:
            self.Growth = Growth
        else:
            if Sigma is not None:
                self.Growth = Sigma.Growth
            else:
                self.Growth = GrowthClass(redshift_limits, redshift_bin_number,
                                          Cosmology=self.Cosmology)

        if Sigma is not None:
            self.Sigma = Sigma
        else:
            self.Sigma = MassVarianceClass(
                self.Growth, mass_limits, mass_bin_number, redshift_limits, redshift_bin_number,
                Cosmology=self.Cosmology)

        self.log_sigmas = self.Sigma.log_sigmas[::-1]
        self.masses = self.Sigma.masses[::-1]
        self.ms_log_sigmas_spl = interpolate.splrep(self.log_sigmas, self.masses)

    def value(self, log_sigma, redshifts=0):
        '''
        Scale to value at redshifts.

        Parameters
        ----------
        log_sigma : float : sigma (no growth function, so z = 0 value)
        redshifts : float or array
        '''
        if redshifts > 0:
            log_sigma -= np.log10(self.Growth.value(redshifts))

        return interpolate.splev(log_sigma, self.ms_log_sigmas_spl)

    def derivative(self, log_sigma):
        return interpolate.splev(log_sigma, self.ms_log_sigmas_spl, der=1)


#===================================================================================================
# characteristic non-linear mass scale
#===================================================================================================
class MassNonlinearClass:
    '''
    Get value and derivative of characteristic collapse log mass as a function of redshift.
    '''

    def __init__(self, Mass=None,
                 redshift_limits=REDSHIFT_LIMITS, redshift_bin_number=REDSHIFT_BIN_NUMBER,
                 mass_limits=MASS_LIMITS, mass_bin_number=MASS_BIN_NUMBER, Cosmology=None):
        '''
        Make spline of log(m_char) v redshift.

        Parameters
        ----------
        Mass : class : mass(sigma) class
        redshift_limits : list : min and max limits of redshift
        redshift_bin_number : int : number of spline points within limits
        mass_limits : list : min and max limits of mass
        mass_bin_number : int : number of spline points within limits
        Cosmology : class : cosmology class
        '''
        if Cosmology is not None:
            self.Cosmology = Cosmology
        else:
            if Mass is not None:
                self.Cosmology = Mass.Cosmology
            else:
                self.Cosmology = COSMOLOGY

        if Mass is not None:
            self.Mass = Mass
        else:
            self.Mass = MassClass(
                None, mass_limits, mass_bin_number, redshift_limits, redshift_bin_number,
                Cosmology=self.Cosmology)

        self.redshifts = np.linspace(redshift_limits[0], redshift_limits[1], redshift_bin_number)
        self.m_chars = np.zeros(redshift_bin_number)
        for zi in range(self.redshifts.size):
            self.m_chars[zi] = self.mchar(self.redshifts[zi])
        self.mchar_redshift_spl = interpolate.splrep(self.redshifts, self.m_chars)

    def value(self, redshift):
        return interpolate.splev(redshift, self.mchar_redshift_spl)

    def derivative(self, redshift):
        return interpolate.splev(redshift, self.mchar_redshift_spl, der=1)

    def mass_nonlinear(self, redshift):
        '''
        Get characteristic non-linear mass [log M_sun/h] at redshift.

        Parameters
        ----------
        redshift : float
        '''
        # select collapse threshold
        # spherical collapse - compute barrier where G(z) * sigma = DELTA_C
        #barrier = DELTA_C / self.Growth.value(redshift)
        barrier = DELTA_C
        mass_char = self.Mass.value(np.log10(barrier), redshift)

        return mass_char

        '''
        # ellipsodial collapse (SMT2001)
        a_smt = 1
        beta = 0.47
        gamma = 0.615
        # ellipsodial collapse (SMT01) updated by Dejacques 08
        #a_smt = 1
        #beta = 0.412
        #gamma = 0.618
        # SMT01 fit to simulation (b=0.2)
        #a_smt = 0.707
        #beta = 0.5
        #gamma = 0.6

        sig = np.arange(5, 0.2, -0.01)
        for i in range(1, len(sig)):
            b = (a_smt)**0.5 * DELTA_C * (1 + beta * (sig[i]**2 / (a_smt * DELTA_C**2))**gamma)
            if sig[i] < b:
                print sig[i]
                break
        #barrier = sig[i] / self.Growth.value(redshift)
        barrier = sig[i]
        mass_char = self.Mass.value(np.log10(barrier))
        return mass_char
        '''

    def print_mchar(self):
        '''
        Print log M_nonlinear at redshift values of spline.
        '''
        for zi in range(self.redshifts.size):
            redshift = self.Mass.Growth.redshifts[zi]
            mass_char = self.mass_nonlinear(redshift)
            print('z {:.4f}  log-mass_nl {:.2f}'.format(redshift, mass_char))


#===================================================================================================
# mass function
#===================================================================================================
def get_dndm(Sigma, mass, redshift, fit_kind='sheth-tormen'):
    '''
    Get integrand d(number-density)/d(mass) [M_sun/h].

    Parameters
    ----------
    Sigma : class : sigma(mass) class
    mass : float : log mass [M_sun/h]
    redshift : float
    fit_kind : str : sheth-tormen, jenkins, warren, tinker
    '''
    density_m = Sigma.Cosmology.get_density('matter', redshift, 'kpc/h comoving')
    log_sigma = Sigma.value(mass, redshift)
    sigma = 10 ** log_sigma
    # find dln(nu)/dlnM = -dln(sigma)/dlnM = -dlog(sigma)/dlogM
    dlnnu_dlnm = -Sigma.derivative(mass)

    # Sheth-Tormen (1999 or 2001)
    if fit_kind == 'sheth-tormen':
        # fit to b=0.2
        a_smt = 0.707
        # ellipoidal collapse
        #a_smt = 1
        q = 0.3
        A = 0.3222
        nu = DELTA_C / 10 ** log_sigma
        f_nu = (A * (2 * a_smt / np.pi) ** 0.5 * (1 + (a_smt * nu ** 2) ** (-q)) *
                np.exp(-0.5 * a_smt * nu ** 2))
        f_nu *= nu  # different normalization for this fit
    # Jenkins et al 2001 (not good below ~3e10 M_sun/h)
    elif fit_kind == 'jenkins':
        A = 0.315
        b = 0.61
        c = 3.8
        f_nu = A * np.exp(-abs(np.log(1 / sigma) + b) ** c)
    # Warren et al 2006 (b = 0.2, with evaporation)
    elif fit_kind == 'warren':
        A = 0.7234
        a = 1.625
        b = 0.2538
        c = 1.1982
        f_nu = A * (sigma ** (-a) + b) * np.exp(-c / sigma ** 2)
    # Tinker et al 2008 (SO(200m))
    elif fit_kind == 'tinker':
        A = 0.186
        a = 1.47
        b = 2.57
        c = 1.19
        f_nu = A * ((sigma / b) ** (-a) + 1) * np.exp(-c / sigma ** 2)

    return density_m / 10 ** (2 * mass) * f_nu * dlnnu_dlnm


def get_mass_function(
    Sigma=None, mass_limits=[11.1, 16.1], mass_width=0.2, redshift=0, fit_kind='tinker',
    Cosmology=COSMOLOGY):
    '''
    Get mass bins [M_sun/h] and d(number_density)/d(mass).

    Parameters
    ----------
    Sigma : class : sigma(mass) class
    mass_limits : list : min and max limits on mass [log M_sun/h]
    mass_width : float : width of mass bin
    redshift : float
    fit_kind : str : sheth-tormen, jenkins, warren, tinker
    Cosmology : class : cosmology class
    '''
    if Sigma is None:
        Sigma = MassVarianceClass(Cosmology=Cosmology)
    masses = np.arange(mass_limits[0], mass_limits[1], mass_width)
    mass_bin_number = masses.size
    mass_function = np.zeros(mass_bin_number)
    for mi in range(mass_bin_number):
        get_dn_dm = get_dndm(Sigma, masses[mi], redshift, fit_kind)
        dn_dlogm = get_dn_dm * 10 ** masses[mi] / np.log10(np.e)
        mass_function[mi] = dn_dlogm

    return masses, np.log10(mass_function)


def print_mass_function(
    Sigma=None, mass_limits=[10, 15], mass_width=0.1, redshift=0, version='tinker',
    Cosmology=COSMOLOGY):
    '''
    Print mass function, d(number-density)/d(log mass) v log mass [M_sun/h].

    Parameters
    ----------
    Sigma : class : sigma(mass) class
    mass_limits : list : min and max limits on log mass
    mass_width : float : width of mass bin
    redshift : float
    version : str : fit version: sheth-tormen, jenkins, warren, tinker
    Cosmology : class : cosmology class
    '''
    if not Sigma:
        Sigma = MassVarianceClass(Cosmology=Cosmology)
    masses = ut.array.get_arange_safe(mass_limits, mass_width)
    mass_bin_number = masses.size
    yplot = np.zeros(mass_bin_number)
    for mi in range(mass_bin_number):
        get_dn_dm = get_dndm(Sigma, masses[mi], redshift, version)
        #dn_dlogm = get_dn_dm * 10**masses[a] / np.log10(e)
        yplot[mi] = np.log10(get_dn_dm * 10 ** (2 * masses[mi]) /
                          (Sigma.Cosmology.get_density('matter', redshift, 'kpc/h comoving')))
        #yplot[mi] = np.log10(get_dn_dm * 10**masses[mi])
        print('log(mass) {:5.2f} | d(num-den)/d(log mass) {:.3e}'.format(masses[mi], get_dn_dm))


#===================================================================================================
# test
#===================================================================================================
def test(redshift=0):
    '''
    Test all of the classes in this module.

    Parameters
    ----------
    redshift : float
    '''
    # calculate growth_function(z) and make spline
    Growth = GrowthClass(Cosmology=COSMOLOGY)
    # calculate sigma(mass), normalized to sigma_8, and make spline
    Sigma = MassVarianceClass(Growth)
    Mass = MassClass(Sigma)
    # calculate M_star(z) and make spline
    MassChar = MassNonlinearClass(Mass)

    #plot_dndm(Sigma)

    for redshift in np.arange(0, REDSHIFT_LIMITS[1] + 0.1, 0.25):
        print('{:.2f} {:.2f}'.format(redshift, MassChar.mass_nonlinear(redshift)))
