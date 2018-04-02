import numpy as np

import defaultparams.params as params
import scipy
import defaultparams.uconv as uconv

from density_models import *
from gen import *

'''
Mass models for DM, gas, stars
'''


def nfw_mass_model(r, c, rs, z):

    '''
    Calculates the NFW profile mass.


    Args:
    -----
    r (float or array) [kpc]: array of radius values

    c (float) [unitless]: mass concentration
    rs (float) [kpc]: scale radius
    z (float): redshift of cluster

    Returns:
    --------
    M (float or array) [Msun]: mass within radius (or radii),
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
    # M = 4.*np.pi*rho_crit*delta_char*(rs**3.)*func_x  # [kg]
    M = 4.*np.pi*rho_crit*delta_char*(rs**3.)*func_x/uconv.Msun  # [Msun]

    return M


def sersic_mass_model(x, normsersic, clustermeta):

    '''
    Calculates the stellar mass of the cluster central galaxy according to the
        3D density profile of the form of the deprojected Sersic profile
        (Lima Neto+ 1999. Eq 20).

    Args:
    -----
    x (array) [kpc]: array of radius values
    normsersic (float): log(normalization [Msun kpc^-3]) of Sersic profile
    clustermeta (dictionary): dictionary of cluster and analysis info produced
        by set_prof_data()


    Returns:
    --------
    M [Msun] (float or array): central galaxy stellar mass within radius (or radii)


    References:
    -----------
    Lima Neto, G. B., Gerbal, D., & Marquez, I. 1999, MNRAS, 309, 481

    '''

    nu = clustermeta['bcg_sersic_n']**-1.

    p = 1.-(0.6097*nu)+(0.00563*(nu**2.))  # limaneto1999
    a = clustermeta['bcg_re']*np.exp(-((0.6950-np.log(nu))/nu)-0.1789)
    f = np.exp(-(((0.6950-np.log(nu))/nu)-0.1789))

    return (4*np.pi*(clustermeta['bcg_re']**3.)*(f**3.)*(10.**normsersic)/nu) \
        * scipy.special.gamma((3-p)/nu) \
        * scipy.special.gammainc((3-p)/nu, (f**-nu)*(x/clustermeta['bcg_re'])**nu)
    # [Msun]


def gas_mass_model(x, nemodel):

    '''
    Calculates the ICM gas mass within some radius (or radii) by integrating
        the gas density profile:
            Mgas = \int 4*pi*r^2 rho_gas dr

    Args:
    -----
    x (array) [kpc]: array of radius values
    nemodel (dictionary): dictionary storing the gas density profile model as
        output in fit_density()

    Returns:
    --------
    mgas [Msun] (float or array): ICM gas mass within radius (or radii)
    '''

    if nemodel['type'] == 'single_beta':

        ne0 = nemodel['parvals'][0]  # [cm^-3]
        rc = nemodel['parvals'][1]  # [kpc]
        beta = nemodel['parvals'][2]  # [unitless]

        mgas = (4./3.)*np.pi*(x**3.)*(params.mu_e*uconv.mA/uconv.Msun) \
            * ((ne0*(uconv.cm_kpc**-3.))
               * scipy.special.hyp2f1(3./2., (3./2.)*beta, 5./2., -(x/rc)**2.))
        # [Msun]

    if nemodel['type'] == 'cusped_beta':

        ne0 = nemodel['parvals'][0]  # [cm^-3]
        rc = nemodel['parvals'][1]  # [kpc]
        beta = nemodel['parvals'][2]  # [unitless]
        alpha = nemodel['parvals'][3]  # [unitless]

        mgas = (4./(3.-alpha))*np.pi*(x**3.)*(params.mu_e*uconv.mA/uconv.Msun) \
               * (ne0*(uconv.cm_kpc**-3.))*((x/rc)**-alpha) \
               * scipy.special.hyp2f1((3.-alpha)/2., (3./2.)*beta,
                                      1.+((3.-alpha)/2.), -(x/rc)**2.)
        # [Msun]

    if nemodel['type'] == 'double_beta_tied':

        ne01 = nemodel['parvals'][0]  # [cm^-3]
        rc1 = nemodel['parvals'][1]  # [kpc]
        beta1 = nemodel['parvals'][2]  # [unitless]

        ne02 = nemodel['parvals'][3]  # [cm^-3]
        rc2 = nemodel['parvals'][4]  # [kpc]
        beta2 = beta1  # TIED TO BETA1!!!!

        mgas = (4./3.)*np.pi*(x**3.)*(params.mu_e*uconv.mA/uconv.Msun) \
               * (((ne01*(uconv.cm_kpc**-3.))
                   * scipy.special.hyp2f1(3./2., (3./2.)*beta1,
                                          5./2., -(x/rc1)**2.))
                  + ((ne02*(uconv.cm_kpc**-3.))
                     * scipy.special.hyp2f1(3./2., (3./2.)*beta2,
                                            5./2., -(x/rc2)**2.)))
        # [Msun]

    if nemodel['type'] == 'double_beta':

        ne01 = nemodel['parvals'][0]  # [cm^-3]
        rc1 = nemodel['parvals'][1]  # [kpc]
        beta1 = nemodel['parvals'][2]  # [unitless]

        ne02 = nemodel['parvals'][3]  # [cm^-3]
        rc2 = nemodel['parvals'][4]  # [kpc]
        beta2 = nemodel['parvals'][5]

        mgas = (4./3.)*np.pi*(x**3.)*(params.mu_e*uconv.mA/uconv.Msun) \
               * (((ne01*(uconv.cm_kpc**-3.))
                   * scipy.special.hyp2f1(3./2., (3./2.)*beta1,
                                          5./2., -(x/rc1)**2.))
                  + ((ne02*(uconv.cm_kpc**-3.))
                     * scipy.special.hyp2f1(3./2., (3./2.)*beta2,
                                            5./2., -(x/rc2)**2.)))
        # [Msun]

    return mgas
