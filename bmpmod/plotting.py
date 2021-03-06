import corner

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import mod_temperature

import defaultparams.uconv as uconv
import defaultparams.cosmology as cosmo

import scipy

from mod_gasdensity import *
from mod_mass import *

'''
Plotting functions
'''


def seplog(n):
    '''
    For a float of the form n=fac*10**power, seperates out "fac" and "power".
    Used with the intent of making nice looking annotations on a plot.
    '''
    power = int(np.floor(np.log10(n)))
    fac = n/(10.**power)
    return [fac, power]


def plt_mcmc_freeparam(mcmc_results, samples, sampler, tspec_data,
                       clustermeta):

    '''
    Make a corner plot from the MCMC posterior distribution of
    free-parameter values.

    Args:
    -----
    mcmc_results (array):
    samples (array): posterior MCMC distribution of free-param vals
    tspec_data (astropy table): table containing profile information about
        temperature

    Results:
    --------
    fig1 (plot)
    '''

    matplotlib.rcParams['font.size'] = 9
    matplotlib.rcParams['axes.labelsize'] = 12

    if samples.shape[1] == 3:
        xa = 0.7
    elif samples.shape[1] == 2:
        xa = 0.6

    fig1 = corner.corner(samples,
                         labels=["$c$",
                                 "$R_s$",
                                 r"$\rho_{\star,0,\mathrm{Sersic}}$"])

    chainshape = np.array(sampler.chain).shape

    plt.annotate('Nwalkers, Nsteps = '
                 + str(chainshape[0])
                 + ', '+str(chainshape[1]),
                 (xa, 0.95), xycoords='figure fraction')

    # plt.annotate('Nburnin = '+str(params.Nburnin),
    #             (xa,0.9),xycoords='figure fraction')

    plt.annotate('$r_{\mathrm{ref}}$='
                 + str(int(tspec_data['radius'][clustermeta['refindex']]))
                 + ' kpc', (xa, 0.8), xycoords='figure fraction')

    plt.annotate(r'$c = '+str(np.round(mcmc_results['c'][0], decimals=1))
                 + '_{-'+str(np.round(mcmc_results['c'][2], decimals=2))
                 + '}^{+'+str(np.round(mcmc_results['c'][1], decimals=2))
                 + '}$', (xa, 0.75), xycoords='figure fraction')

    plt.annotate(r'$R_{s} = '+str(np.round(mcmc_results['rs'][0], decimals=1))
                 + '_{-'+str(np.round(mcmc_results['rs'][2],  decimals=1))
                 + '}^{+'+str(np.round(mcmc_results['rs'][1], decimals=1))
                 + '}$ kpc', (xa, 0.7), xycoords='figure fraction')

    ya = 0.7
    if clustermeta['incl_mstar'] == 1:
        ya = 0.65
        plt.annotate(
            r'$log(\rho_{\star,0,\mathrm{Sersic}} [M_{\odot} kpc^{-3}]) = '
            + str(np.round(mcmc_results['normsersic'][0], decimals=1))
            + '_{-'+str(np.round(mcmc_results['normsersic'][2], decimals=2))
            + '}^{+'+str(np.round(mcmc_results['normsersic'][1], decimals=2))
            + '}$', (xa, 0.65), xycoords='figure fraction')

    # print properties of the sampler
    try:
        # check autocorrelation time
        tacor = sampler.acor

        plt.annotate(
            r'$\tau_{\mathrm{acor}}(c)$='+str(int(np.round(tacor[0], 0))),
            (xa, ya-0.1), xycoords='figure fraction')

        plt.annotate(
            r'$\tau_{\mathrm{acor}}(R_s)$='+str(int(np.round(tacor[1], 0))),
            (xa, ya-0.15), xycoords='figure fraction')

        if clustermeta['incl_mstar'] == 1:
            plt.annotate(
                r'$\tau_{\mathrm{acor}}(log(\rho_{\star,0,\mathrm{Sersic}}))$='
                + str(int(np.round(tacor[2], 0))),
                (xa, ya-0.2), xycoords='figure fraction')

    except:
        pass

    return fig1


###########################################################################
###########################################################################
###########################################################################


def plt_summary(ne_data, tspec_data, nemodel, mcmc_results, clustermeta):

    '''
    Make a summary plot containing the gas density profile, temperature
    profile, and mass profile. Annotations for all relevant calculated
    quantities.

    Args:
    -----
    ne_data (astropy table): table containing profile information about
        gas density
    tspec_data (astropy table): table containing profile information about
        temperature
    nemodel (dictionary): info about ne profile fit including param values
        and errors
    mcmc_results (dictionary): values and errors of free-params of MCMC as
        well as quantities calculated from the posterior MCMC distribution


    Results:
    --------
    fig2 (plot):
         subfig 1: plot of observed gas density profile and fitted gas
            density profile
         subfig 2: plot of observed temperature profile and model temperature
            profile
         subfig 3: mass profile of clustermeta - includes total and components
            of DM, stars, gas
    '''

    fig3 = plt.figure(3, (9, 9))
    plt.figure(3)

    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

    plt.suptitle(str(clustermeta['name']))

    '''
    gas density
    '''
    ax1 = fig3.add_subplot(2, 2, 1)

    plt.loglog(ne_data['radius'], ne_data['ne'], 'o', color='#707070',
               markersize=2)

    plt.errorbar(ne_data['radius'], ne_data['ne'],
                 xerr=[ne_data['radius_lowerbound'],
                       ne_data['radius_upperbound']],
                 yerr=ne_data['ne_err'], linestyle='none', color='b')

    plt.xlim(xmin=1)
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    plt.xlabel('r [kpc]')
    plt.ylabel('$n_{e}$ [cm$^{-3}$]')

    plt_densityprof(nemodel=nemodel, ne_data=ne_data, annotations=1)

    '''
    final kT profile with c, rs
    '''

    if clustermeta['incl_mstar'] == 1:
        tfit_arr \
            = mod_temperature.Tmodel_func(
                ne_data=ne_data,
                tspec_data=tspec_data,
                nemodel=nemodel,
                clustermeta=clustermeta,
                c=mcmc_results['c'][0],
                rs=mcmc_results['rs'][0],
                normsersic=mcmc_results['normsersic'][0])

    elif clustermeta['incl_mstar'] == 0:
        tfit_arr \
            = mod_temperature.Tmodel_func(
                ne_data=ne_data,
                tspec_data=tspec_data,
                nemodel=nemodel,
                clustermeta=clustermeta,
                c=mcmc_results['c'][0],
                rs=mcmc_results['rs'][0])

    ax2 = fig3.add_subplot(2, 2, 2)

    plt.semilogx(tspec_data['radius'], tspec_data['tspec'], 'bo')

    plt.errorbar(tspec_data['radius'], tspec_data['tspec'],
                 xerr=[tspec_data['radius_lowerbound'],
                       tspec_data['radius_upperbound']],
                 yerr=[tspec_data['tspec_lowerbound'],
                       tspec_data['tspec_upperbound']],
                 linestyle='none', color='b')

    plt.xlabel('r [kpc]')
    plt.ylabel('kT [keV]')

    plt.annotate('$r_{\mathrm{ref}}$='
                 + str(int(tspec_data['radius'][clustermeta['refindex']]))
                 + ' kpc', (0.05, 0.9), xycoords='axes fraction')

    xmin,xmax=plt.xlim()
    if xmin<1:
        plt.xlim(xmin=1)
    ymin,ymax=plt.ylim()
    plt.ylim(np.floor(ymin),np.ceil(ymax))

    plt.semilogx(tspec_data['radius'], np.array(tfit_arr), 'r-')

    ##########################################################################

    '''
    OVERDENSITY RADIUS: MASS PROFILE
    '''

    ax3 = fig3.add_subplot(2, 2, 3)

    xplot = np.logspace(np.log10(1.), np.log10(900.), 100)

    mass_nfw = nfw_mass_model(xplot,
                              mcmc_results['c'][0],
                              mcmc_results['rs'][0],
                              clustermeta['z'])  # [Msun]

    mass_tot = np.copy(mass_nfw)
    if clustermeta['incl_mstar'] == 1:
        mass_sersic = sersic_mass_model(xplot, mcmc_results['normsersic'][0],
                                        clustermeta)  # Msun
        mass_tot += mass_sersic

    if clustermeta['incl_mgas'] == 1:
        mass_gas = gas_mass_model(xplot, nemodel)  # [Msun]
        mass_tot += mass_gas

    plt.loglog(xplot, mass_tot, 'r-', label='M$_{\mathrm{tot}}$')
    plt.loglog(xplot, mass_nfw, 'b-', label='M$_{\mathrm{DM}}$')

    if clustermeta['incl_mstar'] == 1:
        plt.loglog(xplot, mass_sersic, 'g-', label='M$_{\star}$')

    if clustermeta['incl_mgas'] == 1:
        plt.loglog(xplot, mass_gas, 'y-', label='M$_{\mathrm{gas}}$')

    handles, labels = ax3.get_legend_handles_labels()
    plt.legend(handles, labels, loc=2)

    plt.xlim(xmin=2)
    plt.ylim(ymin=6.*10**10., ymax=10**14.)  # to match g07

    plt.xlabel('r [kpc]')
    plt.ylabel('mass [$M_{\odot}$]')

    plt.annotate(r'$c_{'+str(int(cosmo.overdensity))+'} = '
                 + str(np.round(mcmc_results['c'][0], 1))
                 + '_{-'+str(np.round(mcmc_results['c'][2], 2))
                 + '}^{+'+str(np.round(mcmc_results['c'][1], 2))+'}$',
                 (0.55, 0.45), xycoords='figure fraction')

    plt.annotate(r'$R_{s} = '+str(np.round(mcmc_results['rs'][0], 1))
                 + '_{-'+str(np.round(mcmc_results['rs'][2], 1))
                 + '}^{+'+str(np.round(mcmc_results['rs'][1], 1))+'}$ kpc',
                 (0.55, 0.4), xycoords='figure fraction')

    if clustermeta['incl_mstar'] == 1:
        plt.annotate(
            r'$log(\rho_{\star,0,\mathrm{Sersic}} [M_{\odot} kpc^{-3}]) = '
            + str(np.round(mcmc_results['normsersic'][0], 1))
            + '_{-'+str(np.round(mcmc_results['normsersic'][2], 2))
            + '}^{+'+str(np.round(mcmc_results['normsersic'][1], 2))
            + '}$',
            (0.55, 0.35), xycoords='figure fraction')

        plt.annotate(
            r'$R_{eff}=$'+str(clustermeta['bcg_re'])+' kpc',
            (0.8, 0.45), xycoords='figure fraction')

        plt.annotate(
            r'$n_{\mathrm{Sersic}}$='+str(clustermeta['bcg_sersic_n']),
            (0.8, 0.4), xycoords='figure fraction')

    plt.annotate(
        '$R_{'+str(int(cosmo.overdensity))+'}='
        + str(int(np.round(mcmc_results['rdelta'][0], 0)))
        + '_{-'+str(int(np.round(mcmc_results['rdelta'][2], 0)))
        + '}^{+'+str(int(np.round(mcmc_results['rdelta'][1], 0)))
        + ' }$ kpc',
        (0.55, 0.25), xycoords='figure fraction')

    plt.annotate(
        '$M_{'+str(int(cosmo.overdensity))+'}='
        + str(np.round(seplog(mcmc_results['mdelta'][0])[0], 2))
        + '_{-'+str(np.round(mcmc_results['mdelta'][2]
                             * 10**-seplog(mcmc_results['mdelta'][0])[1], 2))
        + '}^{+'+str(np.round(mcmc_results['mdelta'][1]
                              * 10**-seplog(mcmc_results['mdelta'][0])[1], 2))
        + '} \ 10^{'+str(seplog(mcmc_results['mdelta'][0])[1])
        + '} \ M_{\odot}$',
        (0.55, 0.2), xycoords='figure fraction')

    plt.annotate(
        '$M_{DM}(R_{'+str(int(cosmo.overdensity))+'})='
        + str(np.round(seplog(mcmc_results['mdm'][0])[0], 2))
        + '_{-'+str(np.round(mcmc_results['mdm'][2]
                             * 10**-seplog(mcmc_results['mdm'][0])[1], 2))
        + '}^{+'+str(np.round(mcmc_results['mdm'][1]
                              * 10**-seplog(mcmc_results['mdm'][0])[1], 2))
        + '} \ 10^{'+str(seplog(mcmc_results['mdm'][0])[1])
        + '} \ M_{\odot}$',
        (0.55, 0.15), xycoords='figure fraction')

    if clustermeta['incl_mgas'] == 1:
        plt.annotate(
            '$M_{gas}(R_{'+str(int(cosmo.overdensity))+'})='
            + str(np.round(seplog(mcmc_results['mgas'][0])[0], 2))
            + '_{-'
            + str(np.round(mcmc_results['mgas'][2]
                           * 10**-seplog(mcmc_results['mgas'][0])[1], 2))
            + '}^{+'
            + str(np.round(mcmc_results['mgas'][1]
                           * 10**-seplog(mcmc_results['mgas'][0])[1], 2))
            + '} \ 10^{'+str(seplog(mcmc_results['mgas'][0])[1])
            + '} \ M_{\odot}$',
            (0.55, 0.10), xycoords='figure fraction')

    if clustermeta['incl_mstar'] == 1:
        plt.annotate(
            '$M_{\star}(R_{'+str(int(cosmo.overdensity))+'})='
            + str(np.round(seplog(mcmc_results['mstars'][0])[0], 2))
            + '_{-'
            + str(np.round(mcmc_results['mstars'][2]
                           * 10**-seplog(mcmc_results['mstars'][0])[1], 2))
            + '}^{+'
            + str(np.round(mcmc_results['mstars'][1]
                           * 10**-seplog(mcmc_results['mstars'][0])[1], 2))
            + '} \ 10^{'+str(seplog(mcmc_results['mstars'][0])[1])
            + '} \ M_{\odot}$',
            (0.55, 0.05), xycoords='figure fraction')

    return fig3, ax1, ax2

#############################################################################
#############################################################################
#############################################################################


def plt_densityprof(nemodel, ne_data, annotations=0):

    '''
    Helper function to plot the input gas density profile model.

    Args:
    -----
    nemodel (dictionary): info about ne profile fit including
        param values and errors
    annotations: option to add ne model parameter values and errors to plot

    Results:
    --------
    plt (plot): a plot with annotations of the best-fitting model of the
        gas density profile.

    '''

    # add model to plot
    rplot = np.linspace(1., max(ne_data['radius']), 1000)

    if nemodel['type'] == 'double_beta':

        plt.plot(rplot, doublebetamodel(nemodel['parvals'], rplot), 'r')

        if annotations == 1:
            plt.annotate(
                r'$n_{e,0,1}='+str(np.round(nemodel['parvals'][0], 3))
                + '_{'+str(np.round(nemodel['parmins'][0], 3))
                + '}^{+'+str(np.round(nemodel['parmaxes'][0], 3))
                + '}$ cm$^{-3}$', (0.02, 0.4), xycoords='axes fraction')

            plt.annotate(
                '$r_{c,1}='+str(np.round(nemodel['parvals'][1], 2))
                + '_{'+str(np.round(nemodel['parmins'][1], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][1], decimals=2))
                + '}$ kpc', (0.02, 0.35), xycoords='axes fraction')

            plt.annotate(
                r'$\beta_1='+str(np.round(nemodel['parvals'][2], 2))
                + '_{'+str(np.round(nemodel['parmins'][2], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][2], decimals=2))
                + '}$', (0.02, 0.3), xycoords='axes fraction')

            plt.annotate(
                r'$n_{e,0,2}='+str(np.round(nemodel['parvals'][3], decimals=3))
                + '_{'+str(np.round(nemodel['parmins'][3], decimals=3))
                + '}^{+'+str(np.round(nemodel['parmaxes'][3], decimals=3))
                + '}$ cm$^{-3}$', (0.02, 0.25), xycoords='axes fraction')

            plt.annotate(
                '$r_{c,2}='+str(np.round(nemodel['parvals'][4], decimals=2))
                + '_{'+str(np.round(nemodel['parmins'][4], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][4], decimals=2))
                + '}$ kpc', (0.02, 0.2), xycoords='axes fraction')

            plt.annotate(
                r'$\beta_2='+str(np.round(nemodel['parvals'][5], decimals=2))
                + '_{'+str(np.round(nemodel['parmins'][5], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][5], decimals=2))
                + '}$', (0.02, 0.15), xycoords='axes fraction')

            plt.annotate(
                '$\chi^2_r$='+str(np.round(nemodel['rchisq'], decimals=2)),
                (0.02, 0.05), xycoords='axes fraction')

    if nemodel['type'] == 'double_beta_tied':

        plt.plot(rplot, doublebetamodel_tied(nemodel['parvals'], rplot), 'r')

        if annotations == 1:

            plt.annotate(
                r'$n_{e,0,1}='+str(np.round(nemodel['parvals'][0], 3))
                + '_{'+str(np.round(nemodel['parmins'][0], 3))
                + '}^{+'+str(np.round(nemodel['parmaxes'][0], 3))
                + '}$ cm$^{-3}$', (0.02, 0.4), xycoords='axes fraction')

            plt.annotate(
                '$r_{c,1}='+str(np.round(nemodel['parvals'][1], 2))
                + '_{'+str(np.round(nemodel['parmins'][1], 2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][1], 2))
                + '}$ kpc', (0.02, 0.35), xycoords='axes fraction')

            plt.annotate(
                r'$\beta_1='+str(np.round(nemodel['parvals'][2], 2))
                + '_{'+str(np.round(nemodel['parmins'][2], 2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][2], 2))
                + '}$', (0.02, 0.3), xycoords='axes fraction')

            plt.annotate(
                r'$n_{e,0,2}='+str(np.round(nemodel['parvals'][3], 3))
                + '_{'+str(np.round(nemodel['parmins'][3], 3))
                + '}^{+'+str(np.round(nemodel['parmaxes'][3], 3))
                + '}$ cm$^{-3}$', (0.02, 0.25), xycoords='axes fraction')

            plt.annotate(
                '$r_{c,2}='+str(np.round(nemodel['parvals'][4], 2))
                + '_{'+str(np.round(nemodel['parmins'][4], 2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][4], 2))
                + '}$ kpc', (0.02, 0.2), xycoords='axes fraction')

            plt.annotate(r'$\beta_2=\beta_1$',
                         (0.02, 0.15), xycoords='axes fraction')

            plt.annotate(
                '$\chi^2_r$='+str(np.round(nemodel['rchisq'], 2)),
                (0.02, 0.05), xycoords='axes fraction')

    if nemodel['type'] == 'single_beta':

        plt.plot(rplot, betamodel(nemodel['parvals'], rplot), 'r')

        if annotations == 1:
            plt.annotate(
                r'$n_{e,0}='+str(np.round(nemodel['parvals'][0], decimals=3))
                + '_{'+str(np.round(nemodel['parmins'][0], decimals=3))
                + '}^{+'+str(np.round(nemodel['parmaxes'][0], decimals=3))
                + '}$ cm$^{-3}$', (0.02, 0.25), xycoords='axes fraction')

            plt.annotate(
                '$r_{c}='+str(np.round(nemodel['parvals'][1], decimals=2))
                + '_{'+str(np.round(nemodel['parmins'][1], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][1], decimals=2))
                + '}$ kpc', (0.02, 0.2), xycoords='axes fraction')

            plt.annotate(
                r'$\beta='+str(np.round(nemodel['parvals'][2], decimals=2))
                + '_{'+str(np.round(nemodel['parmins'][2], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][2], decimals=2))
                + '}$', (0.02, 0.15), xycoords='axes fraction')

            plt.annotate(
                '$\chi^2_r$='+str(np.round(nemodel['rchisq'], decimals=2)),
                (0.02, 0.05), xycoords='axes fraction')

    if nemodel['type'] == 'cusped_beta':

        plt.plot(rplot, cuspedbetamodel(nemodel['parvals'], rplot), 'r')

        if annotations == 1:
            plt.annotate(
                r'$n_{e,0}='+str(np.round(nemodel['parvals'][0], decimals=3))
                + '_{'+str(np.round(nemodel['parmins'][0], decimals=3))
                + '}^{+'+str(np.round(nemodel['parmaxes'][0], decimals=3))
                + '}$ cm$^{-3}$', (0.02, 0.3), xycoords='axes fraction')

            plt.annotate(
                '$r_{c}='+str(np.round(nemodel['parvals'][1], decimals=2))
                + '_{'+str(np.round(nemodel['parmins'][1], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][1], decimals=2))
                + '}$ kpc', (0.02, 0.25), xycoords='axes fraction')

            plt.annotate(
                r'$\beta='+str(np.round(nemodel['parvals'][2], decimals=2))
                + '_{'+str(np.round(nemodel['parmins'][2], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][2], decimals=2))
                + '}$', (0.02, 0.2), xycoords='axes fraction')

            plt.annotate(
                r'$\epsilon='+str(np.round(nemodel['parvals'][3], decimals=2))
                + '_{'+str(np.round(nemodel['parmins'][3], decimals=2))
                + '}^{+'+str(np.round(nemodel['parmaxes'][3], decimals=2))
                + '}$', (0.02, 0.15), xycoords='axes fraction')

            plt.annotate(
                '$\chi^2_r$='+str(np.round(nemodel['rchisq'], decimals=2)),
                (0.02, 0.05), xycoords='axes fraction')

    return plt


###########################################################################
###########################################################################
###########################################################################


def plt_summary_nice(ne_data, tspec_data, nemodel, mcmc_results, clustermeta):

    '''
    Make a summary plot containing the gas density profile, temperature
    profile, and mass profile. Annotations for all relevant calculated
    quantities.

    Nice version to go in paper.

    Args:
    -----
    ne_data (astropy table): table containing profile information about
        gas density
    tspec_data (astropy table): table containing profile information about
        temperature
    nemodel (dictionary): info about ne profile fit including param values
        and errors
    mcmc_results (dictionary): values and errors of free-params of MCMC as
        well as quantities calculated from the posterior MCMC distribution


    Results:
    --------
    fig4 (plot):
         subfig 1: plot of observed gas density profile and fitted gas density
             profile
         subfig 2: plot of observed temperature profile and model temperature
             profile
         subfig 3: mass profile of clustermeta - includes total and components
             of DM, stars, gas
    '''

    fig4 = plt.figure(4, (12, 4))
    plt.figure(4)

    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

    '''
    gas density
    '''
    ax1 = fig4.add_subplot(1, 3, 1)

    plt.loglog(ne_data['radius'], ne_data['ne'], 'o', color='#707070',
               markersize=2)

    plt.errorbar(ne_data['radius'], ne_data['ne'],
                 xerr=[ne_data['radius_lowerbound'],
                       ne_data['radius_upperbound']],
                 yerr=ne_data['ne_err'],
                 linestyle='none', color='#707070')

    plt.xlim(xmin=1)
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    plt.xlabel('r [kpc]')
    plt.ylabel('$n_{e}$ [cm$^{-3}$]')

    plt_densityprof(nemodel=nemodel, ne_data=ne_data, annotations=0)

    '''
    final kT profile with c, rs
    '''
    if clustermeta['incl_mstar'] == 1:
        tfit_arr \
            = mod_temperature.Tmodel_func(
                ne_data=ne_data,
                tspec_data=tspec_data,
                nemodel=nemodel,
                clustermeta=clustermeta,
                c=mcmc_results['c'][0],
                rs=mcmc_results['rs'][0],
                normsersic=mcmc_results['normsersic'][0])

    elif clustermeta['incl_mstar'] == 0:
        tfit_arr \
            = mod_temperature.Tmodel_func(
                ne_data=ne_data,
                tspec_data=tspec_data,
                nemodel=nemodel,
                clustermeta=clustermeta,
                c=mcmc_results['c'][0],
                rs=mcmc_results['rs'][0])

    ax2 = fig4.add_subplot(1, 3, 2)

    plt.semilogx(tspec_data['radius'], tspec_data['tspec'], 'bo')

    plt.errorbar(tspec_data['radius'], tspec_data['tspec'],
                 xerr=[tspec_data['radius_lowerbound'],
                       tspec_data['radius_upperbound']],
                 yerr=[tspec_data['tspec_lowerbound'],
                       tspec_data['tspec_upperbound']],
                 linestyle='none', color='b')

    plt.xlabel('r [kpc]')
    plt.ylabel('kT [keV]')

    plt.ylim(0, 4)
    plt.xlim(xmin=1)

    plt.semilogx(tspec_data['radius'], np.array(tfit_arr), 'r-')

    ##########################################################################

    '''
    OVERDENSITY RADIUS: MASS PROFILE
    '''

    ax3 = fig4.add_subplot(1, 3, 3)

    xplot = np.logspace(np.log10(1.), np.log10(900.), 100)

    mass_nfw = nfw_mass_model(xplot,
                              mcmc_results['c'][0],
                              mcmc_results['rs'][0],
                              clustermeta['z'])  # [Msun]

    mass_tot = np.copy(mass_nfw)
    if clustermeta['incl_mstar'] == 1:
        mass_sersic = sersic_mass_model(xplot, mcmc_results['normsersic'][0],
                                        clustermeta)  # Msun
        mass_tot += mass_sersic

    if clustermeta['incl_mgas'] == 1:
        mass_gas = gas_mass_model(xplot, nemodel)  # [Msun]
        mass_tot += mass_gas

    plt.loglog(xplot, mass_tot, 'r-', label='M$_{\mathrm{tot}}$')
    plt.loglog(xplot, mass_nfw, 'b-', label='M$_{\mathrm{DM}}$')

    if clustermeta['incl_mstar'] == 1:
        plt.loglog(xplot, mass_sersic, 'g-', label='M$_{\star}$')

    if clustermeta['incl_mgas'] == 1:
        plt.loglog(xplot, mass_gas, 'y-', label='M$_{\mathrm{gas}}$')

    handles, labels = ax3.get_legend_handles_labels()
    plt.legend(handles, labels, loc=2)

    plt.xlim(xmin=2)
    plt.ylim(ymin=6.*10**10., ymax=10**14.)  # to match g07

    plt.xlabel('r [kpc]')
    plt.ylabel('mass [$M_{\odot}$]')

    return fig4, ax1
