import defaultparams.params as params
import defaultparams.uconv as uconv

import scipy

from mod_mass import *
from mod_gasdensity import *

'''
Temperature profile model
'''

def intmodel(nemodel, c, rs, normsersic, r_arr, clustermeta):

    '''
    Model of the form \rho_gas * M_tot * r^-2; intended to be integrated when
    calculating Tmodel(r). \rho_gas is the model gas electron number density
    profile and M_tot is the total gravitating mass model.

    Args:
    -----
    nemodel (dictionary): dictionary storing the gas density profile model as
        output in fit_density()
    c (float): concentration parameter of NFW profile
    rs (float) [kpc]: scale radius of NFW profile
    normsersic (float): log(normalization [Msun kpc^-3]) of Sersic model for
        stellar mass profile of cluster central galaxy
    r_arr (array) [kpc]: radial position values of temperature profile
    clustermeta (dictionary): dictionary of cluster and analysis info produced
        by set_prof_data()

    Returns:
    --------
    Function to be integrated when solving for Tmodel(r)
    '''

    Mtot = nfw_mass_model(r_arr, c, rs, clustermeta['z'])
    if clustermeta['incl_mstar'] == 1:
        Mtot += sersic_mass_model(r_arr, normsersic, clustermeta)
    if clustermeta['incl_mgas'] == 1:
        Mtot += gas_mass_model(r_arr, nemodel)

    if nemodel['type'] == 'single_beta':
        return (betamodel(nemodel['parvals'], r_arr)
                * (1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * (Mtot) \
            / (r_arr**2.)

    if nemodel['type'] == 'cusped_beta':
        return (cuspedbetamodel(nemodel['parvals'], r_arr)
                * (1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * (Mtot) \
            / (r_arr**2.)

    if nemodel['type'] == 'double_beta':
        return (doublebetamodel(nemodel['parvals'], r_arr)
                * (1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * (Mtot) \
            / (r_arr**2.)

    if nemodel['type'] == 'double_beta_tied':
        return (doublebetamodel_tied(nemodel['parvals'], r_arr)
                * (1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * (Mtot) \
            / (r_arr**2.)




def Tmodel_func(ne_data,
                tspec_data,
                nemodel,
                clustermeta,
                c, rs, normsersic=0):

    '''
    Calculates the non-parametric model fit to the observed temperature
    profile. Tmodel(r) is calculated from Gastaldello+07 Eq. 2, which follows
    from solving the equation of hydrostatic equilibrium for temperature.


    Args:
    -----
    ne_data (astropy table): observed gas density profile
      in the form established by set_prof_data()
    tspec_data (astropy table): observed temperature profile
      in the form established by set_prof_data()

    nemodel (dictionary): dictionary storing the gas density profile model as
        output in fit_density()
    clustermeta (dictionary): dictionary of cluster and analysis info produced
        by set_prof_data()


    c (float): concentration parameter of NFW profile
    rs (float) [kpc]: scale radius of NFW profile
    normsersic (float): log(normalization [Msun kpc^-3]) of Sersic model for
        stellar mass profile of cluster central galaxy


    Returns:
    --------
    tfit_arr (array) [keV]: model temperature profile values. Position of
        model temperature profile is the same as the input tspec_data['radius']

    References:
    -----------
    Gastaldello, F., Buote, D. A., Humphrey, P. J., et al. 2007, ApJ, 669, 158
    '''

    # return Tmodel given param vals

    ne_ref = nemodel['nefit'][clustermeta['refindex']]
    tspec_ref = tspec_data['tspec'][clustermeta['refindex']]

    radius_ref = tspec_data['radius'][clustermeta['refindex']]



    tfit_arr = []  # to hold array of fit temperature
    # iterate over all radii values in profile
    for rr in range(0, len(tspec_data)):

        if rr == clustermeta['refindex']:
            tfit_arr.append(tspec_data['tspec'][rr])
            continue

        radius_selected = tspec_data['radius'][rr]
        ne_selected = nemodel['nefit'][rr]

        intfunc = lambda x: intmodel(nemodel,
                                     c=c,
                                     rs=rs,
                                     normsersic=normsersic,
                                     r_arr=x,
                                     clustermeta=clustermeta)

        finfac_t = ((params.mu*uconv.mA*uconv.G)
                    / (ne_selected*(uconv.cm_m**-3.)))  # [m6 kg-1 s-2]

        tfit_r = (tspec_ref*ne_ref/ne_selected) \
                 - (uconv.joule_kev*finfac_t*(uconv.Msun_kg**2.)
                    * (uconv.kpc_m**-4.)
                    * scipy.integrate.quad(intfunc, radius_ref,
                                           radius_selected)[0])
        # [kev]

        # print scipy.integrate.quad(intfunc,radius_ref,radius_selected)

        tfit_arr.append(tfit_r)

    return tfit_arr
