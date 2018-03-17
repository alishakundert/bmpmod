__requires__ = ['numpy==1.11.2']
import pkg_resources
pkg_resources.require("numpy==1.11.2")
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import astropy
import astropy.table as atpy
from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.table import Column

import sherpa
import sherpa.ui as ui

import scipy
import scipy.integrate
import scipy.optimize as op


import time

import emcee
import corner


import defaultparams.params as params

from massmod.set_prof_data import set_ne, set_tspec, set_cluster

from examples.vikh_prof import vikh_tprof, vikh_neprof, gen_vik_data

import massmod.fit_density as fit_density
import massmod.fit_temperature as fit_temperature
from massmod.posterior_mcmc import *
from massmod.plotting import *

def seplog(n):
    power=int(np.log10(n))
    fac=n/(10.**power)
    return [fac,power]

if __name__ == '__main__':



    clusterID='RXJ1159+5531'
    clusterID='MKW4'

    
    #generate some fake data accoring to profiles in vikhlinin
    cluster, ne_data, tspec_data, nemodel_vikh, tmodel_vikh=gen_vik_data(clusterID=clusterID, N_ne=50, N_temp=10, noise_ne=0.01, noise_temp=0.05, count_mstar=0)

    ########################################################################
    ########################################################################
    #######################################################################

    '''
    what is the best fitting density model
    '''


    nemodeltype=fit_density.find_nemodeltype(ne_data,tspec_data)


    '''
    fit density profile with beta model - py sherpa
    '''
    
    #need to generalize this a lot to remove double betamodel 
    nemodel=fit_density.fitne(ne_data=ne_data,tspec_data=tspec_data,nemodeltype=str(nemodeltype)) #[cm^-3]
    #nemodel=massmod.fit_density(ne_data,tspec_data,nemodeltype='double_beta') #[cm^-3]

    #data reading and processing above
    ##########################################################################
    ######################################################################### 
    ##########################################################################

    '''
    FITTING MASS PROFILE
    '''


    '''
    Maximum likelihood parameter estimation
    '''
    
    ml_results=fit_temperature.fit_ml(ne_data,tspec_data,nemodel,cluster)

    #http://mathworld.wolfram.com/MaximumLikelihood.html, >define own likelihood functoin



    '''
    MCMC output
    '''
    #col1: c, col2:rs, col3: normsersic
    samples=fit_temperature.fit_mcmc(ne_data,tspec_data,nemodel,ml_results,cluster)


    #col1: rdelta, col2, mdelta, col3: mnfw, col4: mdev, col5: mgas
    #multi-threading using joblib
    samples_aux=calc_posterior_mcmc(samples=samples,nemodel=nemodel,cluster=cluster)


    '''
    Calculate MCMC results
    '''
    
    mcmc_results=samples_results(samples,samples_aux,cluster)

    ##########################################################################
    ######################################################################### 
    ##########################################################################


    '''
    Plot the results
    '''


    '''
    Results MCMC - plotting, free params output
    '''
    fig1=plt_mcmc_freeparam(mcmc_results,samples,tspec_data,cluster)



    '''
    Summary plot
    '''
    fig2=plt_summary(ne_data,tspec_data,nemodel,mcmc_results,cluster)

    ax=fig2.add_subplot(2,2,1)
    xplot=np.logspace(np.log10(min(ne_data['radius'])),np.log10(800.),1000)
    plt.loglog(xplot,vikh_neprof(nemodel_vikh,xplot),'k')
    plt.xlim(xmin=min(ne_data['radius']))

    ax=fig2.add_subplot(2,2,2)
    xplot=np.logspace(np.log10(min(tspec_data['radius'])),np.log10(800.),1000)
    plt.semilogx(xplot,vikh_tprof(tmodel_vikh,xplot),'k-')
    #plt.xlim(xmin=min(tspec_data['radius']))
 




    ##########################################################################
    ######################################################################### 
    ##########################################################################


    plt.tight_layout()
    plt.show()


    #print '/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'.pdf'
    #fig1.savefig('/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'_mcmc.pdf',dpi=300,format='PDF',bbox_inches='tight')
    #fig2.savefig('/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'.pdf',dpi=300,format='PDF',bbox_inches='tight')
