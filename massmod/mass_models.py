import numpy as np

import defaultparams.params as params
import scipy
import defaultparams.uconv as uconv

from density_models import *
from gen import *
'''
Mass models
'''


def nfw_mass_model(r, c, rs, z):

    '''
    Calculates the NFW profile mass.


    Args:
    -----
    r (float or array) [kpc]: array of radius values
    c (float) [unitless]: mass concenration
    rs (float) [kpc]: scale radius



    Returns:
    --------
    M (float or array) [kg]: mass within radius (or radii),
    according to the NFW profile


    References:
    ----------
    Navarro, J. F., Frenk, C. S., & White, S. D. M. 1996, ApJ, 462, 563
    Navarro, J. F., Frenk, C. S., & White, S. D. M. 1997, ApJ, 490, 493


    '''

    r = 1.*np.array(r)

    rho_crit = calc_rhocrit(z)

    func_c = np.log(1.+c)-(c/(1.+c))  # [unitless]

    x = r/rs
    func_x = np.log(1.+x)-(x/(1.+x))  # [unitless]

    # characterstic cosmo.overdensity
    Deltavir = params.overdensity
    delta_char = (Deltavir/3.)*((c**3.)/func_c)  # [unitless]
    # nb: removed OmegaM here because of eq 1 ettori2011

    # mass profile
    M = 4.*np.pi*rho_crit*delta_char*(rs**3.)*func_x  # [kg]
    # M=4.*np.pi*rho_crit*delta_char*(rs**3.)*func_x/uconv.Msun #[Msun]

    return M


def sersic_mass_model(x, normsersic, cluster):

    '''
    Calculates the stellar mass according to the 3D density profile of the form
     of the deprojected Sersic profile (Lima Neto+ 1999. Eq 20).

    Args:
    -----
    x (array) [kpc]: array of radius values
    normsersic (float): log(normalization [Msun kpc^-3])

    #in params file:
    #
    #n (float) [unitless]: Sersic index
    #re (float) [kpc]: effective radius
    #


    Returns:
    --------
    M [Msun] (float or array): mass within radius (or radii)


    References:
    -----------
    Lima Neto, G. B., Gerbal, D., & Marquez, I. 1999, MNRAS, 309, 481

    '''

    nu = cluster['bcg_sersic_n']**-1.

    p = 1.-(0.6097*nu)+(0.00563*(nu**2.))  # limaneto1999
    a = cluster['bcg_re']*np.exp(-((0.6950-np.log(nu))/nu)-0.1789)
    f = np.exp(-(((0.6950-np.log(nu))/nu)-0.1789))

    return (4*np.pi*(cluster['bcg_re']**3.)*(f**3.)*(10.**normsersic)/nu) \
        * scipy.special.gamma((3-p)/nu) \
        * scipy.special.gammainc((3-p)/nu, (f**-nu)*(x/cluster['bcg_re'])**nu)


def mgas_intmodel(x, nemodel):


    '''
    Create the varialble that needs to be integrated to calculate the gas mass.
    Intended to be used in the form:
    Mgas = \int mgas_intmodel dr = \int 4 \pi r^2 \rho(r) dr, where \rho(r) is
    the model desribing the gas mass density

    NOTE: must be used in an integral to actually calcualte the gas mass

    Args:
    -----
    x (float OR ARRAY???):  array of radius values

    nemodel_bfp (array): best-fitting paramters of model fit to ne profile
            accessible with nemodel['parvals'] if using fitne function of this
            package


    Returns:
    --------
    model of gas mass to be integrated

    '''

    fac = 4*np.pi*params.mu_e*uconv.mA/uconv.Msun
    # NB:params.mu_e*uconv.mA changes ne to rho_gas

    if nemodel['type'] == 'single_beta':
        return fac*(x**2.)*(uconv.cm_kpc**-3.) \
            * betamodel(nemodel['parvals'], x)  # [Msun kpc2 kpc-3]

    elif nemodel['type'] == 'double_beta':
        return fac*(x**2.)*(uconv.cm_kpc**-3.) \
            * doublebetamodel(nemodel['parvals'], x)  # [Msun kpc2 kpc-3]

    elif nemodel['type'] == 'cusped_beta':
        return fac*(x**2.)*(uconv.cm_kpc**-3.) \
            * cuspedbetamodel(nemodel['parvals'], x)  # [Msun kpc2 kpc-3]

    elif nemodel['type'] == 'double_beta_tied':
        return fac*(x**2.)*(uconv.cm_kpc**-3.) \
            * doublebetamodel_tied(nemodel['parvals'], x)  # [Msun kpc2 kpc-3]
