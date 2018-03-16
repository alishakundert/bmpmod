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

import defaultparams.cosmology as cosmo
import defaultparams.uconv as uconv
import defaultparams.mcmcparams as mcmcparams

from massmod_plotting import plt_mcmc_freeparam, plt_summary, plt_summary_nice

from joblib import Parallel, delayed



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

    if nemodel['type'] == 'single_beta':
        return (betamodel(nemodel['parvals'], r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
                + sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)

    if nemodel['type'] == 'double_beta':
        return (doublebetamodel(nemodel['parvals'], r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
                +sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)

    if nemodel['type'] == 'cusped_beta':
        return (cuspedbetamodel(nemodel['parvals'], r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
                + sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)

    if nemodel['type'] == 'double_beta_tied':
        return (doublebetamodel_tied(nemodel['parvals'], r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.)) \
            * ((nfw_mass_model(r_arr, c, rs, cluster['z'])/uconv.Msun)
                + sersic_mass_model(r_arr, normsersic, cluster)) \
            / (r_arr**2.)


##############################################################################
##############################################################################
##############################################################################

'''
Fitting function for density profile
'''


def fitne(ne_data, nemodeltype, tspec_data=0):

    '''
    Fits gas number density profile according to selected profile model.
     The fit is performed using python sherpa with the Levenberg-Marquardt
     method of minimizing chis-squared .


    Args:
    -----
    ne_data (astropy table): table containing profile information about
         gas denisty of the required format:
            ne_data['radius']: profile radius values
            ne_data['ne']: profile gas density values
            ne_data['ne_err']: error on gas density values

    tspec_data (astropy table): table containg profile information about
         temperature; requires formation of:
            tspec_data['radius']: profile radius values
            tspec_data['tspec']: profile temperature values
            tspec_data['tspec_err']: error on temperature values

    Returns:
    --------
    nemodel (dictionary)
        nemodel['parvals']: parameter values of fitted gas density model
        nemodel['parmins']: lower error bound on parvals
        nemodel['parmaxes']: upper error bound on parvals
        nemodel['rchisq']: reduced chi-squared of fit
        nemodel['nefit']: fit of profile given radius values from tspec_data

    References:
    -----------
    Python-sherpa:    https://github.com/sherpa/


    '''

    # load data
    ui.load_arrays(1,
                   np.array(ne_data['radius']),
                   np.array(ne_data['ne']),
                   np.array(ne_data['ne_err']))

    # set guess and boundaries on params given selected model

    if nemodeltype == 'single_beta':

        # param estimate
        betaguess = 0.6
        rcguess = 20.  # units?????
        ne0guess = max(ne_data['ne'])

        # beta model
        ui.load_user_model(betamodel, "beta1d")
        ui.add_user_pars("beta1d", ["ne0", "rc", "beta"])
        ui.set_source(beta1d)  # creates model
        ui.set_full_model(beta1d)

        # set parameter values
        ui.set_par(beta1d.ne0, ne0guess,
                   min=0,
                   max=10.*max(ne_data['ne']))
        ui.set_par(beta1d.rc, rcguess,
                   min=0.5,
                   max=max(ne_data['radius']))
        ui.set_par(beta1d.beta, betaguess,
                   min=0.1,
                   max=1.)

    if nemodeltype == 'double_beta':

        # param estimate
        ne0guess1 = max(ne_data['ne'])  # [cm^-3]
        rcguess1 = 10.  # [kpc]
        betaguess1 = 0.6

        ne0guess2 = 0.01*max(ne_data['ne'])  # [cm^-3]
        rcguess2 = 100.  # [kpc]
        betaguess2 = 0.6

        # double beta model
        ui.load_user_model(doublebetamodel, "doublebeta1d")
        ui.add_user_pars("doublebeta1d", ["ne01", "rc1", "beta1",
                         "ne02", "rc2", "beta2"])
        ui.set_source(doublebeta1d)  # creates model
        ui.set_full_model(doublebeta1d)

        # set parameter values
        ui.set_par(doublebeta1d.ne01, ne0guess1,
                   min=0.0001*max(ne_data['ne']),
                   max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc1, rcguess1,
                   min=0.,
                   max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta1, betaguess1,
                   min=0.1,
                   max=1.)

        ui.set_par(doublebeta1d.ne02, ne0guess2,
                   min=0.0001*max(ne_data['ne']),
                   max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc2, rcguess2,
                   min=10.,
                   max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta2, betaguess2,
                   min=0.1,
                   max=1.)

    if nemodeltype == 'cusped_beta':

        # param estimate
        betaguess = 0.7
        rcguess = 5.  # [kpc]
        ne0guess = max(ne_data['ne'])
        alphaguess = 10.  # ????

        # beta model
        ui.load_user_model(cuspedbetamodel, "cuspedbeta1d")
        ui.add_user_pars("cuspedbeta1d", ["ne0", "rc", "beta", "alpha"])
        ui.set_source(cuspedbeta1d)  # creates model
        ui.set_full_model(cuspedbeta1d)

        # set parameter values
        ui.set_par(cuspedbeta1d.ne0, ne0guess,
                   min=0.001*max(ne_data['ne']),
                   max=10.*max(ne_data['ne']))
        ui.set_par(cuspedbeta1d.rc, rcguess,
                   min=1.,
                   max=max(ne_data['radius']))
        ui.set_par(cuspedbeta1d.beta, betaguess,
                   min=0.1,
                   max=1.)
        ui.set_par(cuspedbeta1d.alpha, alphaguess,
                   min=0.,
                   max=100.)

    if nemodeltype == 'double_beta_tied':

        # param estimate
        ne0guess1 = max(ne_data['ne'])
        rcguess1 = 10.
        betaguess1 = 0.6

        ne0guess2 = 0.01*max(ne_data['ne'])
        rcguess2 = 100. 
        betaguess2 = 0.6

        # double beta model
        ui.load_user_model(doublebetamodel, "doublebeta1d")
        ui.add_user_pars("doublebeta1d",
                         ["ne01", "rc1", "beta1", "ne02",
                          "rc2", "beta2"])
        ui.set_source(doublebeta1d)  # creates model
        ui.set_full_model(doublebeta1d)

        # set parameter values
        ui.set_par(doublebeta1d.ne01, ne0guess1,
                   min=0.0001*max(ne_data['ne']),
                   max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc1, rcguess1,
                   min=0.,
                   max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta1, betaguess1,
                   min=0.1,
                   max=1.)

        ui.set_par(doublebeta1d.ne02, ne0guess2,
                   min=0.0001*max(ne_data['ne']),
                   max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc2, rcguess2,
                   min=10.,
                   max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta2, betaguess2,
                   min=0.1,
                   max=1.)

        # tie beta2=beta1
        ui.set_par(doublebeta1d.beta2, doublebeta1d.beta1)

    # fit model
    ui.fit()

    # fit statistics
    chisq = ui.get_fit_results().statval
    dof = ui.get_fit_results().dof
    rchisq = ui.get_fit_results().rstat

    # error analysis
    ui.set_conf_opt("max_rstat", 1e9)
    ui.conf()

    parvals = np.array(ui.get_conf_results().parvals)
    parmins = np.array(ui.get_conf_results().parmins)
    parmaxes = np.array(ui.get_conf_results().parmaxes)

    parnames = [str(x).split('.')[1] for x in
                list(ui.get_conf_results().parnames)]

    # where errors are stuck on a hard limit, change error to Inf
    if None in list(parmins):
        ind = np.where(parmins == np.array(None))[0]
        parmins[ind] = float('Inf')

    if None in list(parmaxes):
        ind = np.where(parmaxes == np.array(None))[0]
        parmaxes[ind] = float('Inf')

    # set up a dictionary to contain usefule results of fit including: parvals
    # - values of free params; parmins - min bound on error of free params;
    # parmaxes - max bound on error of free params

    nemodel = {}
    nemodel['type'] = nemodeltype
    nemodel['parnames'] = parnames
    nemodel['parvals'] = parvals
    nemodel['parmins'] = parmins
    nemodel['parmaxes'] = parmaxes
    nemodel['chisq'] = chisq
    nemodel['dof'] = dof
    nemodel['rchisq'] = rchisq

    # calculate an array that contains the modeled gas density at the same
    # radii positions as the tspec array and add to nemodel dictionary

    # if tspec_data included, calculate value of ne model at the same radius
    # positions as temperature profile
    if tspec_data != 0:
        if nemodeltype == 'double_beta':
            nefit_arr = doublebetamodel(nemodel['parvals'],
                                        np.array(tspec_data['radius']))
            # [cm-3]

        if nemodeltype == 'single_beta':
            nefit_arr = betamodel(nemodel['parvals'],
                                  np.array(tspec_data['radius']))
            # [cm-3]

        if nemodeltype == 'cusped_beta':
            nefit_arr = cuspedbetamodel(nemodel['parvals'],
                                        np.array(tspec_data['radius']))
            # [cm-3]

        if nemodeltype == 'double_beta_tied':
            nefit_arr = doublebetamodel_tied(nemodel['parvals'],
                                             np.array(tspec_data['radius']))
            # [cm-3]

        nemodel['nefit'] = nefit_arr

    return nemodel


##############################################################################
##############################################################################
##############################################################################


def Tmodel_func(c, rs, normsersic, ne_data, tspec_data, nemodel, cluster):

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

        finfac_t = ((cosmo.mu*uconv.mA*uconv.G)/(ne_selected*(uconv.cm_m**-3.)))  # [m6 kg-1 s-2]

        tfit_r = (tspec_ref*ne_ref/ne_selected) \
            - (uconv.joule_kev*finfac_t*(uconv.Msun_kg**2.)*(uconv.kpc_m**-4.)
               * scipy.integrate.quad(intfunc, radius_ref, radius_selected)[0])
        # [kev]

        # print scipy.integrate.quad(intfunc,radius_ref,radius_selected)

        tfit_arr.append(tfit_r)

    return tfit_arr


def calc_rdelta_p(row, nemodel, cluster):

    '''
    Radius corresponding to the input overdensity,
    i.e. M(Rdelta)/ Vol(Rdelta) = overdensity * rho_crit

    The total mass, DM mass, stellar mass of BCG, and ICM gas mass is then
    computed within this radius (rdelta).



    Args:
    -----
    c (float): mass concentration prameter of NFW profile
    rs (float) [kpc]: scale radius of NFW profile
    normsersic: normalization of stellar mass profile
    nemodel_bfp: parameters describing gas density profile

    Returns:
    --------
    rdelta: radius corresponding to overdensity
    mdelta: total mass wihin rdelta
    mdm: dark matter mass within rdelta
    mstars: stellar mass of central galaxy wihtn rdelta
    mgas: gas mass within rdelta

    '''

    c = row[0]
    rs = row[1]
    normsersic = row[2]

    # rdelta(dm only first)
    rdelta_dm = c*rs
    # this is the radius where the density of dm interior is 500*cosmo.rho_crit
    # within this radius the total mass density will be >500*cosmo.rho_crit,
    # so need a larger radius to get to 500

    # calculate mass density at rdelta_dm
    mass_nfw = nfw_mass_model(rdelta_dm, c, rs, cluster['z'])  # [kg]

    mass_dev = sersic_mass_model(rdelta_dm, normsersic, cluster)*uconv.Msun  # [kg]

    intfunc = lambda x: mgas_intmodel(rdelta_dm, nemodel)
    mass_gas = scipy.integrate.quad(intfunc, 0, rdelta_dm)[0]*uconv.Msun  # [kg]

    mass_tot = mass_nfw+mass_dev+mass_gas

    rho_crit = calc_rhocrit(cluster['z'])
    ratio = (mass_tot/((4./3.)*np.pi*rdelta_dm**3.))/rho_crit

    # now let's step forward to find true rdelta(total mass)
    rdelta_tot = int(rdelta_dm)
    while ratio > cosmo.overdensity:

        rdelta_tot += 1

        mass_nfw = nfw_mass_model(rdelta_tot, c, rs, cluster['z'])  # [kg]

        mass_dev = sersic_mass_model(rdelta_tot, normsersic, cluster)*uconv.Msun  # [kg]

        intfunc = lambda x: mgas_intmodel(rdelta_tot, nemodel)
        mass_gas = scipy.integrate.quad(intfunc, 0, rdelta_tot)[0]*uconv.Msun  # [kg]

        mass_tot = mass_nfw+mass_dev+mass_gas

        rho_crit = calc_rhocrit(cluster['z'])
        ratio = (mass_tot/((4./3.)*np.pi*rdelta_tot**3.))/rho_crit

    return rdelta_tot, mass_tot/uconv.Msun, mass_nfw/uconv.Msun, mass_dev/uconv.Msun, mass_gas/uconv.Msun


def posterior_mcmc(samples, nemodel, cluster, Ncores=mcmcparams.Ncores):

    '''
    Calculate the radius corresponding to the given overdensity i.e. the radius
     corresponding to a mean overdensity that is some factor times the critical
     densiy at the redshift of the cluster. Within this radius, calculate the
     total mass, DM mass, stellar mass, gas mass.


    Args:
    -----
    samples (array): contains the posterior MCMC distribution
            col 0: c
            col 1: rs
            col 2: normsersic

    nemodel_bfp (array): contains parameters for best fitting model to the gas
        density profile


    Returns:
    --------
    samples_aux (array): contains output quantities based on the posterior
        MCMC distribution
            col 0: Rdelta
            col 1: Mdelta
            col 2: M(DM, delta)
            col 3: M(stars, delta)
            col 4: M(gas, delta)


    Notes:
    ------
    Utilizes JOBLIB for multi-threading. Number of cores as given in
        params file.
    JOBLIB: https://pythonhosted.org/joblib/

    '''

    samples_aux = Parallel(n_jobs=Ncores)(
        delayed(calc_rdelta_p)(row, nemodel, cluster) for row in samples)

    return np.array(samples_aux)


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

    c, rs, normsersic = theta

    model = Tmodel_func(c, rs, normsersic, ne_data, tspec_data, nemodel, cluster)

    inv_sigma2 = 1.0/(yerr**2)  # CHECK THIS!!!
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# log-prior
def lnprior(theta,
            c_boundmin=mcmcparams.c_boundmin,
            c_boundmax=mcmcparams.c_boundmax,

            rs_boundmin=mcmcparams.rs_boundmin,
            rs_boundmax=mcmcparams.rs_boundmax,

            normsersic_boundmin=mcmcparams.normsersic_boundmin,
            normsersic_boundmax=mcmcparams.normsersic_boundmax):

    '''
    Ensures that free parameters do not go outside bounds

    Args:
    ----
    theta (array): current values of free-paramters; set by WHICH FUNCTION?
    '''
    c, rs, normsersic = theta

    if c_boundmin < c < c_boundmax \
        and rs_boundmin < rs < rs_boundmax \
            and normsersic_boundmin < normsersic < normsersic_boundmax:

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
           c_guess=mcmcparams.c_guess,
           c_boundmin=mcmcparams.c_boundmin,
           c_boundmax=mcmcparams.c_boundmax,

           rs_guess=mcmcparams.rs_guess,
           rs_boundmin=mcmcparams.rs_boundmin,
           rs_boundmax=mcmcparams.rs_boundmax,

           normsersic_guess=mcmcparams.normsersic_guess,
           normsersic_boundmin=mcmcparams.normsersic_boundmin,
           normsersic_boundmax=mcmcparams.normsersic_boundmax):

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


def fit_mcmc(ne_data, tspec_data, nemodel, ml_results, cluster,
             Ncores=mcmcparams.Ncores,
             Nwalkers=mcmcparams.Nwalkers,
             Nsamples=mcmcparams.Nsamples,
             Nburnin=mcmcparams.Nburnin):

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
    ndim, nwalkers = 3, Nwalkers
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


#############################################################################
#############################################################################
############################################################################


def find_nemodeltype(ne_data, tspec_data):

    '''
    Find the best fitting model to the gas density profile. Options include:
    beta model, cusped beta model, double beta model,
        and double beta model tied.

    Best-fitting model is determined by the lowest reduced chi-squared, as
    determined by levenberg marquardt python sherpa.


    Returns:
    --------
    nemodeltype (string): best fitting model

    '''

    opt_models = ['single_beta', 'cusped_beta', 'double_beta',
                  'double_beta_tied']
    opt_rchisq = []

    for ii in range(0, len(opt_models)):
        nemodel = fitne(ne_data=ne_data, nemodeltype=opt_models[ii],
                        tspec_data=tspec_data)
        opt_rchisq.append(nemodel['rchisq'])

    opt_rchisq = np.array(opt_rchisq)
    ind = np.where(opt_rchisq == min(opt_rchisq))[0][0]

    return opt_models[ind]


#############################################################################
#############################################################################
############################################################################


def write_ne(nemodel, fn):

    combo = str(nemodel['type'])
    for ii in range(0, len(nemodel['parvals'])):

        combo += ' & '

        if (nemodel['parnames'][ii] == 'ne01') | (nemodel['parnames'][ii] == 'ne0'):
            combo += '$'+str(np.round(nemodel['parvals'][ii]*(10**1.), 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii]*(10**1.), 2)) \
                + '}^{+'+str(np.round(nemodel['parmaxes'][ii]*(10**1.), 2))+'}$'
            continue

        if nemodel['parnames'][ii] == 'rc1':
            combo += '$'+str(np.round(nemodel['parvals'][ii], 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii], 2)) \
                +'}^{+'+str(np.round(nemodel['parmaxes'][ii], 2))+'}$'
            continue

        if (nemodel['parnames'][ii] == 'beta1') | (nemodel['parnames'][ii] == 'beta'):
            combo += '$'+str(np.round(nemodel['parvals'][ii], 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii], 2)) \
                + '}^{+'+str(np.round(nemodel['parmaxes'][ii], 2))+'}$'
            continue

        if nemodel['parnames'][ii] == 'ne02':
            combo += '$'+str(np.round(nemodel['parvals'][ii]*(10**3.), 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii]*(10**3.), 2)) \
                + '}^{+'+str(np.round(nemodel['parmaxes'][ii]*(10**3.), 2))+'}$'
            continue

        if nemodel['parnames'][ii] == 'rc2':
            combo += '$'+str(int(np.round(nemodel['parvals'][ii], 0))) \
                + '_{'+str(int(np.round(nemodel['parmins'][ii], 0))) \
                + '}^{+'+str(int(np.round(nemodel['parmaxes'][ii], 0)))+'}$'
            continue

    combo += ' & '+str(np.round(nemodel['chisq'], 1))+'/'+str(nemodel['dof']) \
        + '('+str(np.round(nemodel['rchisq'], 2))+')'

    return combo


def calc_rhocrit(z):

    Hz = cosmo.H0*((cosmo.OmegaL+(cosmo.OmegaM*(1.+z)**3.))**0.5)
    rho_crit = (3.*((Hz*uconv.km_Mpc)**2.)) \
        / (8.*np.pi*(uconv.G*(uconv.m_kpc**3.)))  # [kg kpc^-3]

    return rho_crit
