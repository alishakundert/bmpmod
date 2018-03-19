import numpy as np
import defaultparams.uconv as uconv

'''
Density Models
'''

def betamodel(pars, x):

    '''
    Beta model of the form
        \ne = \ne0 [1 +(r/rc)^{2}]^{-3\beta /2}


    Args:
    -----
    pars (array): parameters of model
            of the form: [ne0, rc, beta]

    x (array) [kpc]: position values at which to calculate model

    Returns:
    --------
    electron number density profile

    References:
    -----------
    Cavaliere, A., & Fusco-Femiano, R. 1976, A&A, 49, 137
    Cavaliere, A., & Fusco-Femiano, R. 1978, A&A, 70, 677

    '''

    ne0 = pars[0]  # [cm3]
    rc = pars[1]  # [kpc]
    beta = pars[2]  # [unitless]

    return (ne0 * ((1.+((x/rc)**2.))**((-3.*beta)/2.)))  # [cm^-3]


def cuspedbetamodel(pars, x):

    '''
    Cusped beta model of the form
        \ne = \ne0 [(r/rc)^{-\alpha}]*[1 +(r/rc)^{2}]^{(-3\beta /2)+(\alpha /2)}

    See Humphrey+09 Eq. A1

    Args:
    -----
    pars (array): parameters of model
            of the form: [ne0, rc, beta, epsilon]

    x (array) [kpc]: position values at which to calculate model

    Returns:
    --------
    electron number density profile

    References:
    -----------
    Humphrey, P. J., Buote, D. A., Brighenti, F., Gebhardt, K.,
         & Mathews, W. G. 2009, ApJ, 703, 1257

    '''

    ne0 = pars[0]  # [cm^-3]
    rc = pars[1]  # [kpc]
    beta = pars[2]  # [unitless]
    alpha = pars[3]  # [unitless]

    return ne0*((x/rc)**(-alpha)) \
        * ((1.+((x/rc)**2.))**((-3.*beta/2.)+(alpha/2.)))  # [cm^-3]


def doublebetamodel(pars, x):

    '''
    double beta model of the form
        \ne1 = \ne01 [1 +(r/rc1)^{2}]^{-3\beta1 /2}
        \ne2 =  \ne02 [1 +(r/rc2)^{2}]^{-3\beta2 /2}
        \ne = sqrt(ne1^2 + ne2^2)

    See Humphrey+09 Eq. A2


    Args:
    -----
    pars (array): parameters of model
            of the form: [ne01, rc1, beta1, ne02, rc2, beta2]

    x (array) [kpc]: position values at which to calculate model

    Returns:
    --------
    electron number density profile

    References:
    -----------
    Humphrey, P. J., Buote, D. A., Brighenti, F., Gebhardt, K.,
         & Mathews, W. G. 2009, ApJ, 703, 1257

    '''

    ne01 = pars[0]  # [cm^-3]
    rc1 = pars[1]  # [kpc]
    beta1 = pars[2]  # [unitless]

    ne02 = pars[3]  # [cm^-3]
    rc2 = pars[4]  # [kpc]
    beta2 = pars[5]  # [unitless]

#    return (((ne01**2.) * ((1.+((x/rc1)**2.))**(-3.*beta1)))
#            + ((ne02**2.) * ((1.+((x/rc2)**2.))**(-3.*beta2))))**0.5

    return (ne01 * ((1.+((x/rc1)**2.))**(-3.*beta1/2.))) + (ne02 * ((1.+((x/rc2)**2.))**(-3.*beta2/2.)))


def doublebetamodel_tied(pars, x):

    '''
    double beta model of the form
        \ne1 = \ne01 [1 +(r/rc1)^{2}]^{-3\beta1 /2}
        \ne2 =  \ne02 [1 +(r/rc2)^{2}]^{-3\beta2 /2}
        \ne = sqrt(ne1^2 + ne2^2)

    See Humphrey+09 Eq. A2

    With beta1 = beta2. Both beta values are the same and tied together.


    Args:
    -----
    pars (array): parameters of model
            of the form: [ne01, rc1, beta1, ne02, rc2, beta2]

    x (array) [kpc]: position values at which to calculate model

    Returns:
    --------
    electron number density profile

    References:
    -----------
    Humphrey, P. J., Buote, D. A., Brighenti, F., Gebhardt, K.,
         & Mathews, W. G. 2009, ApJ, 703, 1257

    '''

    ne01 = pars[0]  # [cm^-3]
    rc1 = pars[1]  # [kpc]
    beta1 = pars[2]  # [unitless]

    ne02 = pars[3]  # [cm^-3]
    rc2 = pars[4]  # [kpc]
    beta2 = beta1  # TIED TO BETA1!!!!

#    return (((ne01**2.) * ((1.+((x/rc1)**2.))**(-3.*beta1)))
#            + ((ne02**2.) * ((1.+((x/rc2)**2.))**(-3.*beta2))))**0.5

    return (ne01 * ((1.+((x/rc1)**2.))**(-3.*beta1/2.))) + (ne02 * ((1.+((x/rc2)**2.))**(-3.*beta2/2.)))
