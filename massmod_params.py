import numpy as np

import astropy
import astropy.table as atpy

import massmod_uconv as uconv


obsID='4964'


'''
store paramters for massmod here
'''

#Reference index for T(r); where model T(r) = observed T(r)
refindex=6


#Effective radius of central galaxy
re=11.72 #[kpc], eventually add to FGsample

#Sersic index of central galaxy
sersic_n=2.7 #eventually add to FGsample



#redshift
z=None
'''
params to be read in from table
'''
FGdat=atpy.Table.read('../FG_sample.txt',format='ascii') 
ind=np.where(FGdat['obsID']==int(obsID))[0][0]
kpc_arcsec=FGdat['kpc_arcsec'][ind]
z=FGdat['z'][ind]


'''
gas composition
'''
#assume fully ionized gas
H_massfraction=0.75
He_massfraction=0.25


'''
cosmology
'''
OmegaM=0.3
OmegaL=0.7
H0=70. #[km/s/Mpc]
overdensity=500.


'''
interactive method parameters
'''
nemodel_type='double_beta_tied'
#'cusped_beta'
#'single_beta'
#'double_beta'
#'double_beta_tied'




'''
mcmc input
'''
c_guess=5.
rs_guess=100.
normsersic_guess=10.

c_boundmin=1.
c_boundmax=20.
rs_boundmin=10.
rs_boundmax=150.
normsersic_boundmin=0.1
normsersic_boundmax=100.


Ncores=3
Nwalkers=30
Nsamples=30
Nburnin=10

########################################################################
########################################################################
########################################################################


'''
various calculations as a result of input
'''

#molecular weight things

#molecular weight per free electron
mu_e=(H_massfraction+(0.5*He_massfraction))**-1. #1.176

#total mean molecular weight of ions
mu_I=(H_massfraction+(He_massfraction/4.))**-1.

#total mean molecular weight
mu=((1./mu_I)+(1./mu_e))**-1.
#mu=4./(3.+(5.*H_massfraction)) #approximation
#0.6 for x=0.73,or x=0.75. ettori+13 has mu~0.6
#0.615 for x=0.7

#ne, nH ratio
ne_over_np=1./(mu_e*H_massfraction) 
#1.21 if x=0.7, y=0.3
#1.17 if x=0.73, y=0.25 > cosmic abundance levels --NEED TO CHECK!!!
np_over_ne=mu_e*H_massfraction


#calculate HZ and rho_crit
Hz=H0*((OmegaL+(OmegaM*(1.+z)**3.))**0.5)
rho_crit=(3.*((Hz*uconv.km_Mpc)**2.))/(8.*np.pi*(uconv.G*(uconv.m_kpc**3.)))  #[kg kpc^-3], should be rho_crit at redshift of cluster





