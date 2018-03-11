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


import massmod_func as massmod
import massmod_params as params
import massmod_uconv as uconv
from set_prof_data import set_ne, set_tspec
from examples.vikh_prof import vikh_tprof, vikh_neprof


def seplog(n):
    power=int(np.log10(n))
    fac=n/(10.**power)
    return [fac,power]

if __name__ == '__main__':


    table_ne=atpy.Table.read('./examples/table2_ne.txt',format='ascii')   
    table_kt=atpy.Table.read('./examples/table3_kt.txt',format='ascii')  


    clusterID='RXJ1159+5531'
    clusterID='A133'

    '''
    generate mock ne profile
    according to density model in viklinin2006
    '''

    #radial positions of profile
    rpos_ne=np.logspace(np.log10(1.),np.log10(800.),50) #[kpc]


    #parameter of ne profile in vikh table 2
    ind=np.where(table_ne['cluster']==clusterID)[0][0]
    nemodel_params=table_ne[ind]

    #ne profile of vikh
    ne_true=vikh_neprof(nemodel_params,rpos_ne)

    #now add some errors to the yvalues
    noise=0.01 #percent 1 sigma 

    #want to draw from a gaussian centered on ypos, with sigma=percent noise. output is the final y values, sigma is defined by noise
    ne=np.random.normal(ne_true, noise*ne_true)

    ne_err=noise*ne_true

    #set up proper ne_data table strucuture
    ne_data=set_ne(
        radius=rpos_ne,
        ne=ne,
        ne_err=ne_err)

    ########################################################################
    ########################################################################
    #######################################################################


    '''
    generate mock temperature profile
    according to temperature model in viklinin2006
    '''

    #nb: fewer temperature data points from spectral analysis than density points from surface brightness analysis

    
    rpos_tspec=np.logspace(np.log10(10.),np.log10(800.),15)




    #parameter of temperature profile in vikh table 3
    ind=np.where(table_kt['cluster']==clusterID)[0][0]
    tprof_params=table_kt[ind]

    #temp profile of vikh
    tspec_true=vikh_tprof(tprof_params,rpos_tspec)

    noise=0.05
    
    #add this to make larger errors on outer points and smaller errors on inner points
    noise_fac=np.sqrt(rpos_tspec)/max(np.sqrt(rpos_tspec))


    tspec_err=noise*tspec_true

    tspec=np.random.normal(tspec_true, tspec_err)

    #tspec_err=tspec*noise*noise_fac

    tspec_data=set_tspec(
        radius=rpos_tspec,
        tspec=tspec,
        tspec_err=tspec_err)

    


    ########################################################################
    ########################################################################
    #######################################################################


    '''
    what is the best fitting density model
    '''


    nemodeltype=massmod.find_nemodeltype(ne_data,tspec_data)



    '''
    fit density profile with beta model - py sherpa
    '''
    
    #need to generalize this a lot to remove double betamodel 
    nemodel=massmod.fitne(ne_data,tspec_data,nemodeltype=nemodeltype) #[cm^-3]
    #nemodel=massmod.fitne(ne_data,tspec_data,nemodeltype='double_beta') #[cm^-3]


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
    
    ml_results=massmod.fit_ml(ne_data,tspec_data,nemodel)

    #http://mathworld.wolfram.com/MaximumLikelihood.html, >define own likelihood functoin


    '''
    MCMC output
    '''
    #col1: c, col2:rs, col3: normsersic
    samples=massmod.fit_mcmc(ne_data,tspec_data,nemodel,ml_results)


    #col1: rdelta, col2, mdelta, col3: mnfw, col4: mdev, col5: mgas
    #multi-threading using joblib
    samples_aux=massmod.posterior_mcmc(samples,nemodel)


    '''
    Calculate MCMC results
    '''
    
    c_mcmc, rs_mcmc, normsersic_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

    rdelta_mcmc, mdelta_mcmc, mdm_mcmc, mstars_mcmc, mgas_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples_aux, [16, 50, 84],axis=0)))


    print 'MCMC results'
    print 'MCMC: c=',c_mcmc
    print 'MCMC: rs=',rs_mcmc
    print 'MCMC: normsersic=',normsersic_mcmc





    mcmc_results={'c':c_mcmc,'rs':rs_mcmc,'normsersic':normsersic_mcmc,'rdelta':rdelta_mcmc,'mdelta':mdelta_mcmc,'mdm':mdm_mcmc,'mstars':mstars_mcmc,'mgas':mgas_mcmc}

    ##########################################################################
    ######################################################################### 
    ##########################################################################


    '''
    Plot the results
    '''


    '''
    Results MCMC - plotting, free params output
    '''
    fig1=massmod.plt_mcmc_freeparam(mcmc_results,samples,tspec_data)



    '''
    Summary plot
    '''
    fig2=massmod.plt_summary(ne_data,tspec_data,nemodel,mcmc_results)

    ax=fig2.add_subplot(2,2,1)
    xplot=np.logspace(np.log10(1.),np.log10(800.),1000)
    plt.loglog(xplot,vikh_neprof(nemodel_params,xplot),'k')


    ax=fig2.add_subplot(2,2,2)
    xplot=np.logspace(np.log10(5.),np.log10(800.),1000)
    plt.semilogx(xplot,vikh_tprof(tprof_params,xplot),'k-')
 
    ##########################################################################
    ######################################################################### 
    ##########################################################################


    plt.tight_layout()
    plt.show()

    #print '/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'.pdf'
    #fig1.savefig('/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'_mcmc.pdf',dpi=300,format='PDF',bbox_inches='tight')
    #fig2.savefig('/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'.pdf',dpi=300,format='PDF',bbox_inches='tight')
