import numpy as np

import defaultparams.params as params

import astropy.table as atpy

from massmod.set_prof_data import set_ne, set_tspec, set_meta

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


'''
for mock data testing
'''


def vikhlinin_tprof(pars, r_arr):

    '''
    Temperature profile from Vikhlinin+06 Eq. 4-6 and Table 3. Used for
    example purposes.

    Args:
    -----
    pars (array): parameters of temperature model exactly as from vikh
    r_arr (array): radii values at which to calculate values of
        temperature profile

    Returns:
    --------
    T3D: temperature profile for given model params in Vikhlinin+06 and
        input radius values

    References:
    -----------
    Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691
    '''

    T0 = pars['t0']  # [kev]
    rt = pars['rt']*10**3.  # [kpc]
    a = pars['a']
    b = pars['b']
    c = pars['c']
    Tmin = pars['tmin/t0']*pars['t0']  # [kev]

    rcool = pars['rcool']  # [kpc]
    acool = pars['alphacool']

    x = (r_arr/rcool)**acool
    tcool = (x+(Tmin/T0))/(x+1.)

    t = ((r_arr/rt)**-a)/((1.+((r_arr/rt)**b))**(c/b))

    T3D = T0*tcool*t

    return T3D


def vikhlinin_neprof(pars, r):

    '''
    Electron number density profile from Vikhlinin+06 Eq. 3 and Table 2.
    Used for example purposes.

    Args:
    -----
    pars (array): parameters of temperature model exactly as from vikh
    r_arr (array): radii values at which to calculate values of
        temperature profile

    Returns:
    --------
    ne(r): electron number density  profile for given model params in
        Vikhlinin+06 and input radius values

    References:
    -----------
    Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691
    '''

    rdet = pars['rdet']  # [kpc]
    n0 = pars['n0']*10**-3.  # [cm^-3]
    rc = pars['rc']  # [kpc]
    rs = pars['rs']  # [kpc]
    alpha = pars['alpha']
    beta = pars['beta']
    epsilon = pars['epsilon']
    n02 = pars['n02']*10**-1.  # [cm^-3]
    rc2 = pars['rc2']
    beta2 = pars['beta2']

    gamma = 3.

    if n02 > 0:
        ne = np.sqrt(
            params.ne_over_np
            * (((n0**2.)*((r/rc)**-alpha)
                / (((1.+(r/rc)**2.)**((3.*beta)-(alpha/2.)))
                   * ((1.+(r/rs)**gamma)**(epsilon/gamma))))
               + ((n02**2.)/((1.+(r/rc2)**2.)**(3.*beta2)))))
    else:
        ne = np.sqrt(
            params.ne_over_np
            * (((n0**2.)*((r/rc)**-alpha)
                / (((1.+(r/rc)**2.)**((3.*beta)-(alpha/2.)))
                   * ((1.+(r/rs)**gamma)**(epsilon/gamma))))))

    return np.array(ne)

    #######################################################################
    #######################################################################
    #######################################################################


def gen_mock_data(clusterID,
                  N_ne=50,
                  N_temp=10,
                  noise_ne=0.01,
                  noise_temp=0.05,
                  refindex=-1,
                  incl_mstar=0,
                  incl_mgas=1):

    table_z = atpy.Table.read('./exampledata/table1_z.txt', format='ascii')
    table_ne = atpy.Table.read('./exampledata/table2_ne.txt', format='ascii')
    table_kt = atpy.Table.read('./exampledata/table3_kt.txt', format='ascii')

    table_reff = atpy.Table.read('./exampledata/table_bcg_reff.txt',
                                 format='ascii')

    # redshift of cluster
    z = table_z['z'][np.where(table_z['cluster'] == clusterID)[0][0]]

    # minimum radius of vikhlinin temp prof
    rmin = table_z['rmin_kpc'][np.where(table_z['cluster'] == clusterID)[0][0]]
    rmax = table_z['rdet_kpc'][np.where(table_z['cluster'] == clusterID)[0][0]]

    # set up cosmology
    astropycosmo = FlatLambdaCDM(H0=params.H0 * u.km/u.s/u.Mpc,
                                 Om0=params.OmegaM)

    skyscale = float(astropycosmo.kpc_proper_per_arcmin(z)/60.
                     * u.arcmin/u.kpc)  # [kpc/arcsec]

    # effective radius of bcg
    reff_arcsec \
        = table_reff['2MASS_K_Reff_arcsec'][
            np.where(table_reff['cluster'] == clusterID)[0][0]]  # [arcsec]
    reff = np.round(reff_arcsec*skyscale, decimals=2)  # [kpc]

    cluster = set_meta(name=clusterID,
                       z=z,
                       bcg_re=reff,
                       bcg_sersic_n=4.,
                       refindex=refindex,
                       incl_mstar=incl_mstar,
                       incl_mgas=incl_mgas)

    '''
    generate mock ne profile
    according to density model in viklinin2006
    '''

    # radial positions of profile
    rpos_ne = np.logspace(np.log10(rmin), np.log10(rmax), N_ne)  # [kpc]

    # error bars on rpos
    radius_lowerbound = []
    radius_upperbound = []
    for ii in range(0, len(rpos_ne)):

        if ii == 0:
            radius_lowerbound.append((rpos_ne[ii]-rmin)/2.)
            radius_upperbound.append((rpos_ne[ii+1]-rpos_ne[ii])/2.)
            continue

        if ii == len(rpos_ne)-1:
            radius_upperbound.append((rmax-rpos_ne[ii])/2.)
            radius_lowerbound.append((rpos_ne[ii]-rpos_ne[ii-1])/2.)
            continue

        radius_lowerbound.append((rpos_ne[ii]-rpos_ne[ii-1])/2.)
        radius_upperbound.append((rpos_ne[ii+1]-rpos_ne[ii])/2.)

    # parameter of ne profile in vikh table 2
    ind = np.where(table_ne['cluster'] == clusterID)[0][0]
    nemodel_params = table_ne[ind]

    # ne profile of vikhlinin
    ne_true = vikhlinin_neprof(nemodel_params, rpos_ne)

    # want to draw from a gaussian centered on ypos, with sigma=percent noise.
    # output is the final y values, sigma is defined by noise
    ne = np.random.normal(ne_true, noise_ne*ne_true)

    ne_err = noise_ne*ne_true

    # set up proper ne_data table strucuture
    ne_data = set_ne(radius=np.array(rpos_ne),
                     ne=np.array(ne),
                     ne_err=np.array(ne_err),
                     radius_lowerbound=np.array(radius_lowerbound),
                     radius_upperbound=np.array(radius_upperbound))

    #######################################################################

    '''
    generate mock temperature profile
    according to temperature model in viklinin2006
    '''

    # nb: fewer temperature data points from spectral analysis than density
    # points from surface brightness analysis

    rpos_tspec = np.logspace(np.log10(rmin), np.log10(rmax), N_temp)

    # error bars on rpos
    radius_lowerbound = []
    radius_upperbound = []
    for ii in range(0, len(rpos_tspec)):

        if ii == 0:
            radius_lowerbound.append((rpos_tspec[ii]-rmin)/2.)
            radius_upperbound.append((rpos_tspec[ii+1]-rpos_tspec[ii])/2.)
            continue

        if ii == len(rpos_tspec)-1:
            radius_upperbound.append((rmax-rpos_tspec[ii])/2.)
            radius_lowerbound.append((rpos_tspec[ii]-rpos_tspec[ii-1])/2.)
            continue

        radius_lowerbound.append((rpos_tspec[ii]-rpos_tspec[ii-1])/2.)
        radius_upperbound.append((rpos_tspec[ii+1]-rpos_tspec[ii])/2.)

    # parameter of temperature profile in vikh table 3
    ind = np.where(table_kt['cluster'] == clusterID)[0][0]
    tprof_params = table_kt[ind]

    # temp profile of vikhlinin
    tspec_true = vikhlinin_tprof(tprof_params, rpos_tspec)

    # add this to make larger errors on outer points and smaller errors on
    # inner points
    noise_fac = np.sqrt(rpos_tspec)/max(np.sqrt(rpos_tspec))

    tspec_err = noise_temp*tspec_true

    tspec = np.random.normal(tspec_true, tspec_err)

    # tspec_err=tspec*noise*noise_fac

    tspec_data = set_tspec(radius=np.array(rpos_tspec),
                           tspec=np.array(tspec),
                           tspec_err=np.array(tspec_err),
                           radius_lowerbound=np.array(radius_lowerbound),
                           radius_upperbound=np.array(radius_upperbound))

    return cluster, ne_data, tspec_data, nemodel_params, tprof_params
