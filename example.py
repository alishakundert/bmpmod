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


#suppress log info from sherpa
import logging
logger = logging.getLogger("sherpa")
logger.setLevel(logging.ERROR)


#default parameters and unit conversion factors
import defaultparams.params as params
import defaultparams.uconv as uconv

#functions to read data into format used by module
from massmod.set_prof_data import set_ne, set_tspec, set_meta

#function to fit the gas density profile
from massmod.fit_density import fitne, find_nemodeltype

#function to determine mass profile through backwards modelling
from massmod.fit_temperature import fit_ml, fit_mcmc

#analyze the marginalized posterior distribution
from massmod.posterior_mcmc import calc_posterior_mcmc, samples_results

#plotting functions
from massmod.plotting import plt_mcmc_freeparam, plt_summary, plt_summary_nice

#functions specifically to generate mock data from Vikhlinin+ profiles
from exampledata.vikh_prof import vikh_tprof, vikh_neprof, gen_vik_data

if __name__ == '__main__':


    '''
    Goal:

    The primary goal of this example script is to showcase the tools available 
    in the massmod package using mock data. The mock data is produced by 
    randomly sampling the density and temperature profiles models published in 
    Vikhlinin+06 for a sample of clusters. A secondary goal of this example is 
    thus to also explore how the backwards mass modeling process used in the 
    massmod package compares to the forward fitting results of Vikhlinin+06.


    The mock profiles allow for a flexible choice in noise and radial sampling 
    rate, which allows for exploration of how these quantities affect the 
    output of the backwards-fitting process. There is also some flexibility 
    built into the massmod package that can be additionally tested here such 
    as allowing for the stellar mass of the central galaxy to be included (or 
    not included) in the model of total gravitating mass. If the stellar mass 
    profile of the BCG is toggled on, the values for the BCG effective radius 
    Re are pulled from the 2MASS catalog values for a de Vaucouleurs fit to 
    their K-band data.

    
    After generating the mock temperature and density profiles, the below code 
    walks the user through fitting a model to the gas density profile, and 
    performing the backwards-fitting mass modelling analysis. The output 
    includes a non-parametric model fit to the temperature profile, the total 
    mass profile and its associated parameters describing the profile, and the 
    contributions of different mass components (i.e., DM, stars, gas) to the 
    total mass profile.


    ######

    A note on usage:

    Any of the clusters in Vikhlinin+06 are options to be used to generate 
    randomly sampled temperature and density profiles. The full list of 
    clusters is as follows:
    
    vikhlinin_clusters=[A133,
                        A262,
                        A383,
                        A478,
                        A907,
                        A1413,
                        A1795,
                        A1991,
                        A2029,
                        A2390,
                        RXJ1159+5531,
                        MKW4,   
                        USGCS152]  

    After selecting one of these clusters, this example script will 
    automatically generate the cluster and profile data in the proper format 
    to be used by the module. If you have your own data you would like to 
    analyze with the massmod package, please see the included template.py file 
    for instructions.

    '''
    ########################################################################
    ########################################################################
    ########################################################################


    #select any cluster ID from the Vikhlinin+ paper
    clusterID='A262'

    

    '''
    Generate mock gas density and temperature profiles - according to the 
    modeles in Vikhlinin+06. Some pertinent details are included below, more 
    details are included in the docstrings of the functions.

    Args:
    -----
    N_ne: the number of gas density profile data points 
    N_temp: the number of temperature profile
    noise_ne: the percent noise on the density values
    noise_temp: the percent noise on the temperature values
    incl_mstar: include stellar mass of the central galaxy in the model for 
                total gravitating mass
    incl_mgas: include gas mass of ICM in the model for total gravitating mass

    Returns:
    --------
    clustermeta: dictionary that stores relevant properties of cluster 
                 (e.g. redshift) as well as selections for analysis 
                 (e.g., incl_mstar, incl_mgas)
    ne_data: dictionary that stores the mock "observed" gas density profile
    tspec_data: dictionary that store the mock "observed" temperature profile
    nemodel_vikh: parameters of Vikhlinin+06 density model 
    tmodel_vikh: parameters of Vikhlinin+06 temperature model

    
    '''
    clustermeta, ne_data, tspec_data, nemodel_vikh, tmodel_vikh= \
            gen_vik_data(clusterID=clusterID, 
                         N_ne=50,  
                         N_temp=10,  
                         noise_ne=0.01, 
                         noise_temp=0.05,  
                         incl_mstar=1, 
                         incl_mgas=1) 

 
    ########################################################################
    ########################################################################
    #######################################################################

    '''
    Gas density profile
    '''

    #Determine the best fitting model to the density profile. Output will be one of the following: 'single_beta', 'cusped_beta', 'double_beta', 'double_beta_tied'
    nemodeltype=find_nemodeltype(ne_data=ne_data,
                                 tspec_data=tspec_data)



    #Find the parameters and param errors of the best-fitting gas density model
    nemodel=fitne(ne_data=ne_data,
                  tspec_data=tspec_data,
                  nemodeltype=str(nemodeltype)) #[cm^-3]



    ##########################################################################
    ######################################################################### 
    ##########################################################################

    '''
    Maximum likelihood parameter estimation
    
    Perform the backwards-fit of the mass model. The free parameters in the fit are:
    - the mass concentration "c" of the NFW profile used to model the DM halo, 
    - the scale radius "Rs" of the NFW profile
    - optionally, the normalization of the Sersic model "\rho_{\star,0}" used to model the stellar mass profile of the central galaxy
    '''
    
    #estimate the free parameters of the mass model through maximum likelihood
    ml_results=fit_ml(ne_data,tspec_data,nemodel,clustermeta)


    '''
    MCMC parameter estimation

    The backwards-fitting mass modelling process is performed using the MCMC 
    algorithm emcee. The walkers of the ensemble are started from the parameter
    estimation output by the maximum likelihood analysis. Note the number of 
    cores the MCMC analysis is run on is an option here. 

    **warning: default Ncores=3, 
               default Nwalkers, Nsteps, Nburnin are small numbers to allow for fast testing

    Returns:
    --------
    samples - the marginalized posterior distribution
    sampler - the sampler class output by emcee

    '''
    #fit for the mass model and temperature profile model through MCMC
    samples, sampler = fit_mcmc(ne_data=ne_data, 
                                tspec_data=tspec_data, 
                                nemodel=nemodel, 
                                ml_results=ml_results, 
                                clustermeta=clustermeta,
                                Ncores=params.Ncores,
                                Nwalkers=params.Nwalkers,
                                Nsteps=params.Nsteps,
                                Nburnin=params.Nburnin)

    #analyze the marginalized MCMC distribution to calculate Rdelta, Mdelta
    samples_aux = calc_posterior_mcmc(samples=samples, 
                                      nemodel=nemodel, 
                                      clustermeta=clustermeta)


    #summary of the MCMC results
    mcmc_results=samples_results(samples=samples,
                                 samples_aux=samples_aux,
                                 clustermeta=clustermeta)

    ##########################################################################
    ######################################################################### 
    ##########################################################################


    '''
    Plot the results
    '''

    #Corner plot of marginalized posterior distribution of free params from MCMC
    fig1 = plt_mcmc_freeparam(mcmc_results, samples, sampler, tspec_data, clustermeta)



    #Summary plot: density profile, temperature profile, mass profile
    fig2 = plt_summary(ne_data, tspec_data, nemodel, mcmc_results, clustermeta)

    ax=fig2.add_subplot(2,2,1)
    xplot=np.logspace(np.log10(min(ne_data['radius'])),np.log10(800.),1000)
    plt.loglog(xplot,vikh_neprof(nemodel_vikh,xplot),'k')
    plt.xlim(xmin=min(ne_data['radius']))

    ax=fig2.add_subplot(2,2,2)
    xplot=np.logspace(np.log10(min(tspec_data['radius'])),np.log10(800.),1000)
    plt.semilogx(xplot,vikh_tprof(tmodel_vikh,xplot),'k-')
    #plt.xlim(xmin=min(tspec_data['radius']))
 


    plt.tight_layout()
    plt.show()



