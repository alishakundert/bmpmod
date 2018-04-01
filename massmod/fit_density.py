import sherpa.ui as ui
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from density_models import *
import plotting


def find_nemodeltype(ne_data, tspec_data, optplt=0):

    '''
    Find the best fitting model to the gas density profile. Options include:
    beta model, cusped beta model, double beta model,
        and double beta model tied.

    Best-fitting model is determined by the lowest reduced chi-squared, as
    determined by levenberg marquardt python sherpa.

    Args:
    -----
    optplt(int): option to plot


    Returns:
    --------
    nemodeltype (string): best fitting model

    '''

    opt_models = ['single_beta', 'cusped_beta', 'double_beta_tied', 'double_beta']
    opt_rchisq = []

    if optplt==1:
        fig1 = plt.figure(1, (8, 8))
        fig1.clf()

        maxy=0
        miny=999
        
    for ii in range(0, len(opt_models)):
        nemodel = fitne(ne_data=ne_data, nemodeltype=opt_models[ii],
                        tspec_data=tspec_data)
        opt_rchisq.append(nemodel['rchisq'])


        if optplt==1:
            
            if ii==0:
                ax0=fig1.add_subplot(2,2,ii+1)
                ax0.set_xscale("log", nonposx='clip')
                ax0.set_yscale("log", nonposy='clip')
            if ii==1:
                ax1=fig1.add_subplot(2,2,ii+1)
                ax1.set_xscale("log", nonposx='clip')
                ax1.set_yscale("log", nonposy='clip')                
            if ii==2:
                ax2=fig1.add_subplot(2,2,ii+1)
                ax2.set_xscale("log", nonposx='clip')
                ax2.set_yscale("log", nonposy='clip')               
            if ii==3:
                ax3=fig1.add_subplot(2,2,ii+1)
                ax3.set_xscale("log", nonposx='clip')
                ax3.set_yscale("log", nonposy='clip')

            #best-fitting density model
            plotting.plt_densityprof(nemodel,annotations=1)

            #data
            plt.errorbar(ne_data['radius'], ne_data['ne'],
                         xerr=[ne_data['radius_lowerbound'], 
                               ne_data['radius_upperbound']],
                         yerr=ne_data['ne_err'], 
                         marker='o', markersize=2,
                         linestyle='none', color='b')

            plt.annotate(str(opt_models[ii]),(0.55,0.9),xycoords='axes fraction')


            plt.xlabel('r [kpc]')
            plt.ylabel('$n_{e}$ [cm$^{-3}$]')
            
            ymin,ymax=plt.ylim()
            if ymax>maxy:
                maxy=ymax
            if ymin<miny:
                miny=ymin

    
    if optplt==1:
        
        ax0.set_ylim(miny,maxy)
        ax1.set_ylim(miny,maxy)
        ax2.set_ylim(miny,maxy)
        ax3.set_ylim(miny,maxy)

        plt.tight_layout()

    opt_rchisq = np.array(opt_rchisq)
    ind = np.where(opt_rchisq == min(opt_rchisq))[0][0]

    return opt_models[ind], fig1

#############################################################################
#############################################################################
############################################################################




'''
Fitting function for density profile
'''


def fitne(ne_data, nemodeltype, tspec_data=None):

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

    #remove any existing models and data
    ui.clean()
    
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
                   min=0.1,
                   max=max(ne_data['radius']))
        ui.set_par(beta1d.beta, betaguess,
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
                   min=0.1,
                   max=max(ne_data['radius']))
        ui.set_par(cuspedbeta1d.beta, betaguess,
                   min=0.1,
                   max=1.)
        ui.set_par(cuspedbeta1d.alpha, alphaguess,
                   min=0.,
                   max=100.)

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
                   min=0.1,
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


    if nemodeltype == 'double_beta_tied':

        # param estimate
        ne0guess1 = max(ne_data['ne'])
        rcguess1 = 10.
        betaguess1 = 0.6

        ne0guess2 = 0.01*max(ne_data['ne'])
        rcguess2 = 100. 


        # double beta model
        ui.load_user_model(doublebetamodel_tied, "doublebeta1d_tied")
        ui.add_user_pars("doublebeta1d_tied",
                         ["ne01", "rc1", "beta1", "ne02",
                          "rc2"])
        ui.set_source(doublebeta1d_tied)  # creates model
        ui.set_full_model(doublebeta1d_tied)

        # set parameter values
        ui.set_par(doublebeta1d_tied.ne01, ne0guess1,
                   min=0.00001*max(ne_data['ne']),
                   max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d_tied.rc1, rcguess1,
                   min=0.1,
                   max=max(ne_data['radius']))
        ui.set_par(doublebeta1d_tied.beta1, betaguess1,
                   min=0.1,
                   max=1.)

        ui.set_par(doublebeta1d_tied.ne02, ne0guess2,
                   min=0.00001*max(ne_data['ne']),
                   max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d_tied.rc2, rcguess2,
                   min=10.,
                   max=max(ne_data['radius']))

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
    if tspec_data is not None:
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

