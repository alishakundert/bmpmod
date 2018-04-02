import numpy as np

import defaultparams.cosmology as cosmo
import defaultparams.uconv as uconv

'''
General functions
'''


def calc_rhocrit(z):

    '''
    Calculate the critical density of the universe.

    Args:
    -----
    z (float): redshift

    Returns:
    --------
    rho_crit (float): critical density at z in units of [kg kpc^-3]
    '''

    Hz = cosmo.H0*((cosmo.OmegaL+(cosmo.OmegaM*(1.+z)**3.))**0.5)
    rho_crit = (3.*((Hz*uconv.km_Mpc)**2.)) \
        / (8.*np.pi*(uconv.G*(uconv.m_kpc**3.)))  # [kg kpc^-3]

    return rho_crit
