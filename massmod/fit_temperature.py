import numpy as np

import astropy
import astropy.table as atpy
from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

import sherpa
import sherpa.ui as ui

import scipy
import scipy.integrate
import scipy.optimize as op

import emcee

import defaultparams.params as params
import defaultparams.uconv as uconv

from joblib import Parallel, delayed

from mass_models import *
from density_models import *


##############################################################################
##############################################################################
##############################################################################


'''
Integration models for total gravitating mass in T(r)
'''


def intmodel(nemodel, rs, c, normsersic, r_arr, cluster):

    '''

    Model of the form \rho_gas * M_tot * r^-2. \rho_gas is decided by the
     selected model to fit the electron number density profile.
     Intended to be integrated when solving for T(r).

    Note: must actually be integrated later to be useful

    Args:
    -----
    nemodel_bfp
    rs (float) [kpc]: scale radius of NFW profile
    c (float): concenration parameter of NFW profile
    normsersic (float): normalization of Sersic model
    r_arr (array) [kpc]: position values of temperature profile

    Returns:
    --------
    Model to be integrated when solving for T(r)

    '''

    if cluster['count_mstar']==0:
        normsersic=0

    if nemodel['type'] == 'single_beta':
        return (betamodel(nemodel['parvals'], r_arr)
                *(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
               + sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)

    if nemodel['type'] == 'double_beta':
        return (doublebetamodel(nemodel['parvals'], r_arr)
                *(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
               +sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)

    if nemodel['type'] == 'cusped_beta':
        return (cuspedbetamodel(nemodel['parvals'], r_arr)
                *(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
               + sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)

    if nemodel['type'] == 'double_beta_tied':
        return (doublebetamodel_tied(nemodel['parvals'], r_arr)
                *(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
               + sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)


##############################################################################
##############################################################################
##############################################################################


def Tmodel_func(ne_data, tspec_data, nemodel, cluster,c, rs, normsersic=0):

    '''
    Calculates the non-parameteric model fit to the observed temperature
    profile. Model T(r) is calculated from Gastaldello+07 Eq. 2, which follows
    from solving the equation of hydrostatic equilibrium for temperature.


    Args:
    -----
    c (float): concentration parameter of NFW profile
    rs (float) [kpc]: scale radius of NFW profile
    normsersic (float) [log(Msun kpc-3)]: log value of Sersic profile model
    ne_data:
    tspec_data:
    nemodel:

    Returns:
    --------
    tfit_arr (array) [keV]: model temperature profile values. Position of
    model temperature profile is the same as the input tspec_data['radius']

    References:
    -----------
    Gastaldello, F., Buote, D. A., Humphrey, P. J., et al. 2007, ApJ, 669, 158
    '''

    # return Tmodel given param vals

    ne_ref = nemodel['nefit'][cluster['refindex']]
    tspec_ref = tspec_data['tspec'][cluster['refindex']]

    radius_ref = tspec_data['radius'][cluster['refindex']]

    tfit_arr = []  # to hold array of fit temperature
    # iterate over all radii values in profile
    for rr in range(0, len(tspec_data)):

        if rr == cluster['refindex']:
            tfit_arr.append(tspec_data['tspec'][rr])
            continue

        radius_selected = tspec_data['radius'][rr]
        ne_selected = nemodel['nefit'][rr]

        intfunc = lambda x: intmodel(nemodel,
                                     rs=rs,
                                     c=c,
                                     normsersic=normsersic,
                                     r_arr=x,
                                     cluster=cluster)

        finfac_t = ((params.mu*uconv.mA*uconv.G)
                    /(ne_selected*(uconv.cm_m**-3.)))  # [m6 kg-1 s-2]

        tfit_r = (tspec_ref*ne_ref/ne_selected) \
            - (uconv.joule_kev*finfac_t*(uconv.Msun_kg**2.)*(uconv.kpc_m**-4.)
               * scipy.integrate.quad(intfunc, radius_ref, radius_selected)[0])
        # [kev]

        # print scipy.integrate.quad(intfunc,radius_ref,radius_selected)

        tfit_arr.append(tfit_r)

    return tfit_arr






##############################################################################
##############################################################################
##############################################################################

'''
MCMC
'''


# define the log likelihood function, here assumes normal distribution
def lnlike(theta, x, y, yerr, ne_data, tspec_data, nemodel, cluster):
    '''
    Log(likelihood function) comparing observed and model temperature
        profile values.

    Args:
    -----
    theta (array): free-parameters of model
                   nominally of the form [c, rs, normsersic]
    x (?) [?]: positions of observed temperature profile
    y (array) [keV]: observed temperature profile temperature values
    yerr (array) [keV]: errors on temperauture profile values
    ne_data
    tspec_data
    nemodel

    Returns:
    --------
    log of the likelood function. Will be a large value when difference
        between observed and model fits is low.

    '''

    if cluster['count_mstar']==1:
        c, rs, normsersic = theta
    else:
        c, rs = theta
        normsersic=0.

    model = Tmodel_func(ne_data=ne_data, tspec_data=tspec_data, nemodel=nemodel, cluster=cluster,c=c, rs=rs, normsersic=normsersic)

    inv_sigma2 = 1.0/(yerr**2)  # CHECK THIS!!!
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# log-prior
def lnprior(theta,
            c_boundmin=params.c_boundmin,
            c_boundmax=params.c_boundmax,

            rs_boundmin=params.rs_boundmin,
            rs_boundmax=params.rs_boundmax,

            normsersic_boundmin=params.normsersic_boundmin,
            normsersic_boundmax=params.normsersic_boundmax):

    '''
    Ensures that free parameters do not go outside bounds

    Args:
    ----
    theta (array): current values of free-paramters; set by WHICH FUNCTION?
    '''
    if len(theta)==3:
        c, rs, normsersic = theta

        if c_boundmin < c < c_boundmax and rs_boundmin < rs < rs_boundmax and normsersic_boundmin < normsersic < normsersic_boundmax:
            return 0.0
        return -np.inf


    elif len(theta)==2:
        c, rs = theta

        if c_boundmin < c < c_boundmax and rs_boundmin < rs < rs_boundmax:
            return 0.0
        return -np.inf


# full-log probability function
def lnprob(theta, x, y, yerr, ne_data, tspec_data, nemodel, cluster):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, ne_data, tspec_data, nemodel,
                       cluster)


def fit_ml(ne_data, tspec_data, nemodel, cluster,
           c_guess=params.c_guess,
           c_boundmin=params.c_boundmin,
           c_boundmax=params.c_boundmax,

           rs_guess=params.rs_guess,
           rs_boundmin=params.rs_boundmin,
           rs_boundmax=params.rs_boundmax,

           normsersic_guess=params.normsersic_guess,
           normsersic_boundmin=params.normsersic_boundmin,
           normsersic_boundmax=params.normsersic_boundmax):

    '''
    Perform maximum likelikhood parameter estimatation. Results can be used
    as initial guess for more intensive MCMC parameter search.

    Args:
    -----
    ne_data
    tspec_data
    nemodel

    Returns:
    --------
    ml_results (array): results of Maximum-likelihood parameter estimation
       of the form [c_ml,rs_ml,normsersic_ml] which are the best-fitting
       resutls of the paramter estimation.


    '''

    nll = lambda *args: -lnlike(*args)


    if cluster['count_mstar']==1:

        result = op.minimize(nll, [c_guess, rs_guess, normsersic_guess],
                             args=(tspec_data['radius'], tspec_data['tspec'],
                                   tspec_data['tspec_err'], ne_data, tspec_data,
                                   nemodel, cluster),
                             bounds=((c_boundmin, c_boundmax),
                                     (rs_boundmin, rs_boundmax),
                                     (normsersic_boundmin, normsersic_boundmax)))
        
        c_ml, rs_ml, normsersic_ml = result["x"]
        print 'scipy.optimize results'
        print 'ML: c=', c_ml
        print 'ML: rs=', rs_ml
        print 'ML: normsersic=', normsersic_ml
        
        return [c_ml, rs_ml, normsersic_ml]


    elif cluster['count_mstar']==0:
        result = op.minimize(nll, [c_guess, rs_guess],
                             args=(tspec_data['radius'], tspec_data['tspec'],
                                   tspec_data['tspec_err'], ne_data, tspec_data,
                                   nemodel, cluster),
                             bounds=((c_boundmin, c_boundmax),
                                     (rs_boundmin, rs_boundmax)))
        
        c_ml, rs_ml = result["x"]
        print 'scipy.optimize results'
        print 'ML: c=', c_ml
        print 'ML: rs=', rs_ml
        
        return [c_ml, rs_ml]



def fit_mcmc(ne_data, tspec_data, nemodel, ml_results, cluster,
             Ncores=params.Ncores,
             Nwalkers=params.Nwalkers,
             Nsamples=params.Nsamples,
             Nburnin=params.Nburnin):

    '''
    Run MCMC on the free parameters of model for total gravitating mass.
    Utilizes EMCEE.


    Args:
    -----
    ne_data (astropy table): observed gas density profile  (see Notes)
    tspec_data (astropy table): observed temperature profile  (see Notes)
    nemodel (dictionary): best-fitting model to observed gas denisty profile (
        see Notes)
    ml_results (array): maximum-likelihood paramter estimation for free params
            of the form [c_ml, rs_ml, normsersic_ml]

    Returns:
    --------
    samples (array): MCMC sampler chain (??? - i don't know what this means)
            of the form:
                col 1: c
                col 2: rs
                col 3: normsersic
            NB: length of samples array set by Nwalkers * Nsamples

    References:
    -----------
    EMCEE

    '''

    # initialize walkers - result comes from ML fit before

    if cluster['count_mstar']==1:
        ndim, nwalkers = 3, Nwalkers
    elif cluster['count_mstar']==0:
        ndim, nwalkers = 2, Nwalkers

    pos = [ml_results + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    # sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(tspec_data['radius'],
                                          tspec_data['tspec'],
                                          tspec_data['tspec_err'],
                                          ne_data,
                                          tspec_data, nemodel, cluster),
                                    threads=Ncores)
    # WHY ARE THE ARGS THE WAY THEY ARE???

    # run mcmc for 500 steps
    sampler.run_mcmc(pos, Nsamples)
    samples = sampler.chain[:, Nburnin:, :].reshape((-1, ndim))
    # length of samples = walkers*steps

    return samples


