import numpy as np

import defaultparams.params as params

import astropy.table as atpy

from massmod.set_prof_data import set_ne, set_tspec, set_cluster

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


'''
for mock data testing
'''


def vikh_tprof(pars, r_arr):

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


def vikh_neprof(pars, r):

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

    ne = np.sqrt(params.ne_over_np*(((n0**2.)*((r/rc)**-alpha)
            / (((1.+(r/rc)**2.)**((3.*beta)-(alpha/2.)))
                * ((1.+(r/rs)**gamma)**(epsilon/gamma))))
            + ((n02**2.)/((1.+(r/rc2)**2.)**(3.*beta2)))))

    return ne


    #######################################################################
    #######################################################################
    #######################################################################


def gen_vik_data(clusterID, N_ne=50, N_temp=10, noise_ne=0.01, noise_temp=0.05, count_mstar=0):

    table_z=atpy.Table.read('./examples/table1_z.txt',format='ascii') 
    table_ne=atpy.Table.read('./examples/table2_ne.txt',format='ascii')   
    table_kt=atpy.Table.read('./examples/table3_kt.txt',format='ascii')  

    table_reff=atpy.Table.read('./examples/table_bcg_reff.txt',format='ascii')

    #redshift of cluster
    z= table_z['z'][np.where(table_z['cluster']==clusterID)[0][0]]

    #minimum radius of vikhlinin temp prof
    rmin=table_z['rmin_kpc'][np.where(table_z['cluster']==clusterID)[0][0]]


    
    # set up cosmology
    astropycosmo = FlatLambdaCDM(H0=params.H0 * u.km/u.s/u.Mpc, Om0=params.OmegaM)
    skyscale = float(astropycosmo.kpc_proper_per_arcmin(z)/60. \
        * u.arcmin/u.kpc)  # [kpc/arcsec]


    #effective radius of bcg
    reff_arcsec=table_reff['2MASS_K_Reff_arcsec'][np.where(table_reff['cluster']==clusterID)[0][0]] #[arcsec]
    reff=reff_arcsec*skyscale #[kpc]

    cluster=set_cluster(name=clusterID, z=z, bcg_re=reff, bcg_sersic_n=4., refindex=-1, count_mstar=count_mstar)


    '''
    generate mock ne profile
    according to density model in viklinin2006
    '''

    #radial positions of profile
    rpos_ne=np.logspace(np.log10(rmin),np.log10(800.),N_ne) #[kpc]


    #parameter of ne profile in vikh table 2
    ind=np.where(table_ne['cluster']==clusterID)[0][0]
    nemodel_params=table_ne[ind]

    #ne profile of vikh
    ne_true=vikh_neprof(nemodel_params,rpos_ne)


    #want to draw from a gaussian centered on ypos, with sigma=percent noise. output is the final y values, sigma is defined by noise
    ne=np.random.normal(ne_true, noise_ne*ne_true)

    ne_err=noise_ne*ne_true

    #set up proper ne_data table strucuture
    ne_data=set_ne(
        radius=rpos_ne,
        ne=ne,
        ne_err=ne_err)

    #######################################################################


    '''
    generate mock temperature profile
    according to temperature model in viklinin2006
    '''

    #nb: fewer temperature data points from spectral analysis than density points from surface brightness analysis

    
    rpos_tspec=np.logspace(np.log10(rmin),np.log10(800.),N_temp)




    #parameter of temperature profile in vikh table 3
    ind=np.where(table_kt['cluster']==clusterID)[0][0]
    tprof_params=table_kt[ind]

    #temp profile of vikh
    tspec_true=vikh_tprof(tprof_params,rpos_tspec)

    
    #add this to make larger errors on outer points and smaller errors on inner points
    noise_fac=np.sqrt(rpos_tspec)/max(np.sqrt(rpos_tspec))


    tspec_err=noise_temp*tspec_true

    tspec=np.random.normal(tspec_true, tspec_err)

    #tspec_err=tspec*noise*noise_fac

    tspec_data=set_tspec(
        radius=rpos_tspec,
        tspec=tspec,
        tspec_err=tspec_err)


    return cluster, ne_data, tspec_data, nemodel_params, tprof_params
