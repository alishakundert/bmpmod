import numpy as np
import massmod_params as params

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
