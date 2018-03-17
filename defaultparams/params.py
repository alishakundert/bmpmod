import cosmology as cosmo

from cosmology import *
from mcmcparams import *


def mu(X,Y):

    '''
    calculate molecular weight
    '''

    # molecular weight per free electron
    mu_e = (X+(0.5*Y))**-1.  # 1.176

    # total mean molecular weight of ions
    mu_I = (X+(Y/4.))**-1.
    
    # total mean molecular weight
    mu = ((1./mu_I)+(1./mu_e))**-1.
    # mu=4./(3.+(5.*X)) #approximation
    # 0.6 for x=0.73,or x=0.75. ettori+13 has mu~0.6
    # 0.615 for x=0.7

    # ne, nH ratio
    ne_over_np = 1./(mu_e*X)
    # 1.21 if x=0.7, y=0.3
    # 1.17 if x=0.73, y=0.25 > cosmic abundance levels --NEED TO CHECK!!!
    np_over_ne = mu_e*X

    return mu, mu_e, ne_over_np



mu, mu_e, ne_over_np = mu(cosmo.H_massfraction, cosmo.He_massfraction)
