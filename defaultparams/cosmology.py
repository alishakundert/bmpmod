import numpy as np
import uconv as uconv

'''
cosmology
'''
OmegaM = 0.3
OmegaL = 0.7
H0 = 70.  # [km/s/Mpc]
overdensity = 500.

'''
gas composition
'''
# assume fully ionized gas
H_massfraction = 0.75
He_massfraction = 0.25


#########################################################################
#########################################################################
#########################################################################

# put this somewhere else...

# molecular weight things

# molecular weight per free electron
mu_e = (H_massfraction+(0.5*He_massfraction))**-1.  # 1.176

# total mean molecular weight of ions
mu_I = (H_massfraction+(He_massfraction/4.))**-1.

# total mean molecular weight
mu = ((1./mu_I)+(1./mu_e))**-1.
# mu=4./(3.+(5.*H_massfraction)) #approximation
# 0.6 for x=0.73,or x=0.75. ettori+13 has mu~0.6
# 0.615 for x=0.7

# ne, nH ratio
ne_over_np = 1./(mu_e*H_massfraction)
# 1.21 if x=0.7, y=0.3
# 1.17 if x=0.73, y=0.25 > cosmic abundance levels --NEED TO CHECK!!!
np_over_ne = mu_e*H_massfraction
