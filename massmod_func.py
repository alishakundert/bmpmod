import numpy as np

import matplotlib
import matplotlib.pyplot as plt

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


import time

import emcee
import corner


import massmod_params as params
import massmod_uconv as uconv
from massmod_plotting import plt_mcmc_freeparam, plt_summary


import joblib
from joblib import Parallel, delayed



'''
Mass models
'''

def nfw_mass_model(r, c, rs):

    '''
    Calculates the NFW profile mass.


    Args:
    -----
    r (float or array) [kpc]: array of radius values
    c (float) [unitless]: mass concenration
    rs (float) [kpc]: scale radius

 

    Returns:
    --------
    M (float or array) [kg]: mass within radius (or radii), 
    according to the NFW profile


    References:
    ----------
    Navarro, J. F., Frenk, C. S., & White, S. D. M. 1996, ApJ, 462, 563
    Navarro, J. F., Frenk, C. S., & White, S. D. M. 1997, ApJ, 490, 493


    '''
    
    r=1.*np.array(r) 


    func_c=np.log(1.+c)-(c/(1.+c)) #[unitless]

    x=r/rs
    func_x=np.log(1.+x)-(x/(1.+x)) #[unitless]

    #characterstic params.overdensity
    Deltavir=params.overdensity
    delta_char=(Deltavir/3.)*((c**3.)/func_c) #[unitless]
    #nb: removed OmegaM here because of eq 1 ettori2011

    #mass profile
    M=4.*np.pi*params.rho_crit*delta_char*(rs**3.)*func_x #[kg]
    #M=4.*np.pi*params.rho_crit*delta_char*(rs**3.)*func_x/uconv.Msun #[Msun]

    return M 


def sersic_mass_model(x,normsersic):

    '''
    Calculates the stellar mass according to the 3D density profile of the form of the deprojected Sersic profile (Lima Neto+ 1999. Eq 20).

    Args:
    -----
    x (array) [kpc]: array of radius values
    normsersic (float): log(normalization [Msun kpc^-3])

    #in params file:
    #
    #n (float) [unitless]: Sersic index
    #re (float) [kpc]: effective radius
    #


    Returns:
    --------
    M [Msun] (float or array): mass within radius (or radii)


    References:
    -----------
    Lima Neto, G. B., Gerbal, D., & Marquez, I. 1999, MNRAS, 309, 481 

    '''
    
    
    nu=params.sersic_n**-1.

    p=1.-(0.6097*nu)+(0.00563*(nu**2.)) #limaneto1999
    a=params.re*np.exp(-((0.6950-np.log(nu))/nu)-0.1789)
    f=np.exp(-(((0.6950-np.log(nu))/nu)-0.1789))

    
    return (4*np.pi*(params.re**3.)*(f**3.)*(10.**normsersic)/nu)*scipy.special.gamma((3-p)/nu)*scipy.special.gammainc((3-p)/nu,(f**-nu)*(x/params.re)**nu) 



def mgas_intmodel(x,nemodel_bfp):

    '''
    Creates the varialble that needs to be integrated to calculate the gas mass.Intended to be used in the form:  Mgas = \int mgas_intmodel dr = \int 4 \pi r^2 \rho(r) dr, where \rho(r) is the model desribing the gas mass density

    NOTE: must be used in an integral to actually calcualte the gas mass

    Args:
    -----
    x (float OR ARRAY???):  array of radius values

    nemodel_bfp (array): best-fitting paramters of model fit to ne profile
            accessible with nemodel['parvals'] if using fitne function of this package


    Returns:
    --------
    model of gas mass to be integrated

    '''

    fac=4*np.pi*params.mu_e*uconv.mA/uconv.Msun 
    #NB:params.mu_e*uconv.mA changes electron number density to gas mass density

    if params.nemodel_type=='single_beta':
        return fac*(x**2.)*(uconv.cm_kpc**-3.)*betamodel(nemodel_bfp,x) #[Msun kpc2 kpc-3]

    elif params.nemodel_type=='double_beta':
        return fac*(x**2.)*(uconv.cm_kpc**-3.)*doublebetamodel(nemodel_bfp,x) #[Msun kpc2 kpc-3]

    elif params.nemodel_type=='cusped_beta':
        return fac*(x**2.)*(uconv.cm_kpc**-3.)*cuspedbetamodel(nemodel_bfp,x) #[Msun kpc2 kpc-3]

    elif params.nemodel_type=='double_beta_tied':
        return fac*(x**2.)*(uconv.cm_kpc**-3.)*doublebetamodel_tied(nemodel_bfp,x) #[Msun kpc2 kpc-3]

##############################################################################
##############################################################################
##############################################################################

'''
Density Models
'''

def betamodel(pars,x):
    
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

    ne0=pars[0] #[cm3]
    rc=pars[1] #[kpc]
    beta=pars[2] #[unitless]


    return (ne0 * ((1.+((x/rc)**2.))**((-3.*beta)/2.))) #[cm^-3]


def cuspedbetamodel(pars,x):

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
    Humphrey, P. J., Buote, D. A., Brighenti, F., Gebhardt, K., & Mathews, W. G. 2009, ApJ, 703, 1257

    '''
    
    ne0=pars[0] # [cm^-3]
    rc=pars[1] # [kpc]
    beta=pars[2] # [unitless]
    alpha=pars[3] # [unitless]

    
    return ne0*((x/rc)**(-alpha))*((1.+((x/rc)**2.))**((-3.*beta/2.)+(alpha/2.))) #[cm^-3]



def doublebetamodel(pars,x):

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
    Humphrey, P. J., Buote, D. A., Brighenti, F., Gebhardt, K., & Mathews, W. G. 2009, ApJ, 703, 1257

    '''

    ne01=pars[0] # [cm^-3]
    rc1=pars[1] # [kpc]
    beta1=pars[2] # [unitless]

    ne02=pars[3] # [cm^-3]
    rc2=pars[4] # [kpc]
    beta2=pars[5] # [unitless]

    return (((ne01**2.) * ((1.+((x/rc1)**2.))**(-3.*beta1)))+((ne02**2.) * ((1.+((x/rc2)**2.))**(-3.*beta2))))**0.5


def doublebetamodel_tied(pars,x):


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
    Humphrey, P. J., Buote, D. A., Brighenti, F., Gebhardt, K., & Mathews, W. G. 2009, ApJ, 703, 1257

    '''

    
    ne01=pars[0] # [cm^-3]
    rc1=pars[1] # [kpc]
    beta1=pars[2] # [unitless]

    ne02=pars[3] # [cm^-3]
    rc2=pars[4] # [kpc]
    beta2=beta1 #TIED TO BETA1!!!!


    return (((ne01**2.) * ((1.+((x/rc1)**2.))**(-3.*beta1)))+((ne02**2.) * ((1.+((x/rc2)**2.))**(-3.*beta2))))**0.5

##############################################################################
##############################################################################
##############################################################################


'''
Integration models for total gravitating mass in T(r)
'''

def intmodel(nemodel_bfp, rs, c, normsersic, r_arr):

    '''

    Model of the form \rho_gas * M_tot * r^-2. \rho_gas is decided by the selected model to fit the electron number density profile. Intended to be integrated when solving for T(r).

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

    if params.nemodel_type == 'single_beta':
        return (betamodel(nemodel_bfp,r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.))*((nfw_mass_model(r_arr,c,rs)/uconv.Msun)+sersic_mass_model(r_arr,normsersic)) \
            / (r_arr**2.)
    

    if params.nemodel_type == 'double_beta':
        return (doublebetamodel(nemodel_bfp,r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.))*((nfw_mass_model(r_arr,c,rs)/uconv.Msun)+sersic_mass_model(r_arr,normsersic)) \
            / (r_arr**2.)

    if params.nemodel_type == 'cusped_beta':
        return (cuspedbetamodel(nemodel_bfp,r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.))*((nfw_mass_model(r_arr,c,rs,)/uconv.Msun)+sersic_mass_model(r_arr,normsersic)) \
            / (r_arr**2.)

    if params.nemodel_type == 'double_beta_tied':
        return (doublebetamodel_tied(nemodel_bfp,r_arr)*(1./uconv.Msun)*(uconv.cm_kpc**-3.))*((nfw_mass_model(r_arr,c,rs)/uconv.Msun)+sersic_mass_model(r_arr,normsersic)) \
            / (r_arr**2.)



##############################################################################
##############################################################################
##############################################################################

'''
Fitting function for density profile
'''

def fitne(ne_data,tspec_data):

    '''
    
    Fits gas number density profile according to selected profile model. The fit is performed using python sherpa with the Levenberg-Marquardt method of minimizing chis-squared .


    Args:
    -----
    ne_data (astropy table): table containing profile information about gas denisty of the required format:
        ne_data['radius']: profile radius values
        ne_data['ne']: profile gas density values
        ne_data['ne_err']: error on gas density values

    tspec_data (astropy table): table containg profile information about temperature; requires formation of:
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
    
    #load data
    ui.load_arrays(1,np.array(ne_data['radius']),np.array(ne_data['ne']),np.array(ne_data['ne_err'])) 


    #set guess and boundaries on params given selected model

    if params.nemodel_type == 'single_beta':

        #param estimate
        betaguess=0.6
        rcguess=20. #units?????
        S0guess=max(ne_data['ne'])

        #beta model
        ui.load_user_model(betamodel,"beta1d")
        ui.add_user_pars("beta1d",["S0","rc","beta"])
        ui.set_source(beta1d) #creates model
        ui.set_full_model(beta1d)

        #set parameter values
        ui.set_par(beta1d.S0,S0guess,min=0,max=10.*max(ne_data['ne']))
        ui.set_par(beta1d.rc,rcguess,min=0.5,max=max(ne_data['radius']))
        ui.set_par(beta1d.beta,betaguess,min=0.1,max=1.)


    if params.nemodel_type == 'double_beta':

        #param estimate
        S0guess1=max(ne_data['ne']) #[cm^-3]
        rcguess1=10.  #[kpc]
        betaguess1=0.6

        S0guess2=0.01*max(ne_data['ne']) #[cm^-3]
        rcguess2=100. #[kpc]
        betaguess2=0.6


        #double beta model
        ui.load_user_model(doublebetamodel,"doublebeta1d")
        ui.add_user_pars("doublebeta1d",["S01","rc1","beta1","S02","rc2","beta2"])
        ui.set_source(doublebeta1d) #creates model
        ui.set_full_model(doublebeta1d)


        #set parameter values
        ui.set_par(doublebeta1d.S01,S0guess1,min=0.00001*max(ne_data['ne']),max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc1,rcguess1,min=0.,max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta1,betaguess1,min=0.1,max=1.)

        ui.set_par(doublebeta1d.S02,S0guess2,min=0.00001*max(ne_data['ne']),max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc2,rcguess2,min=10.,max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta2,betaguess2,min=0.1,max=1.)


    if params.nemodel_type == 'cusped_beta':

        #param estimate
        betaguess=0.7
        rcguess=5. #[kpc]
        S0guess=max(ne_data['ne']) 
        alphaguess=10. #????

        #beta model
        ui.load_user_model(cuspedbetamodel,"cuspedbeta1d")
        ui.add_user_pars("cuspedbeta1d",["S0","rc","beta","alpha"])
        ui.set_source(cuspedbeta1d) #creates model
        ui.set_full_model(cuspedbeta1d)

        #set parameter values
        ui.set_par(cuspedbeta1d.S0,S0guess,min=0.001*max(ne_data['ne']),max=10.*max(ne_data['ne']))
        ui.set_par(cuspedbeta1d.rc,rcguess,min=1.,max=max(ne_data['radius']))
        ui.set_par(cuspedbeta1d.beta,betaguess,min=0.1,max=1.)
        ui.set_par(cuspedbeta1d.alpha,alphaguess,min=0.,max=100.)

    if params.nemodel_type == 'double_beta_tied':

        #param estimate
        S0guess1=max(ne_data['ne'])
        rcguess1=10. 
        betaguess1=0.6

        S0guess2=0.01*max(ne_data['ne'])
        rcguess2=100. 
        betaguess2=0.6

        #double beta model
        ui.load_user_model(doublebetamodel,"doublebeta1d")
        ui.add_user_pars("doublebeta1d",["S01","rc1","beta1","S02","rc2","beta2"])
        ui.set_source(doublebeta1d) #creates model
        ui.set_full_model(doublebeta1d)


        #set parameter values
        ui.set_par(doublebeta1d.S01,S0guess1,min=0.00001*max(ne_data['ne']),max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc1,rcguess1,min=0.,max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta1,betaguess1,min=0.1,max=1.)

        ui.set_par(doublebeta1d.S02,S0guess2,min=0.00001*max(ne_data['ne']),max=100.*max(ne_data['ne']))
        ui.set_par(doublebeta1d.rc2,rcguess2,min=10.,max=max(ne_data['radius']))
        ui.set_par(doublebeta1d.beta2,betaguess2,min=0.1,max=1.)

        #tie beta2=beta1
        ui.set_par(doublebeta1d.beta2,doublebeta1d.beta1)



    #fit model
    ui.fit()

    #fit statistics
    chisq=ui.get_fit_results().statval
    dof=ui.get_fit_results().dof
    rchisq=ui.get_fit_results().rstat

    #error analysis
    ui.set_conf_opt("max_rstat",1e9)
    ui.conf()

    parvals=np.array(ui.get_conf_results().parvals)
    parmins=np.array(ui.get_conf_results().parmins)
    parmaxes=np.array(ui.get_conf_results().parmaxes)


    #where errors are stuck on a hard limit, change error to Inf instead of None

    ind=np.where(parmins == None)[0]
    parmins[ind]=float('Inf')

    ind=np.where(parmaxes == None)[0]
    parmaxes[ind]=float('Inf')


    
    
    #set up a dictionary to contain usefule results of fit including: parvals - values of free params; parmins - min bound on error of free params; parmaxes - max bound on error of free params 
    nemodel = {'parvals':parvals, 'parmins':parmins, 'parmaxes':parmaxes, 'rchisq':rchisq}


    #calculate an array that contains the modeled gas density at the same radii positions as the tspec array and add to nemodel dictionary

    if params.nemodel_type == 'double_beta':
        nefit_arr=doublebetamodel(nemodel['parvals'],np.array(tspec_data['radius'])) # [cm-3]

    
    if params.nemodel_type == 'single_beta':
        nefit_arr=betamodel(nemodel['parvals'],np.array(tspec_data['radius'])) # [cm-3]


    if params.nemodel_type == 'cusped_beta':
        nefit_arr=cuspedbetamodel(nemodel['parvals'],np.array(tspec_data['radius'])) # [cm-3]


    if params.nemodel_type == 'double_beta_tied':
        nefit_arr=doublebetamodel_tied(nemodel['parvals'],np.array(tspec_data['radius'])) # [cm-3]


    
    nemodel['nefit']=nefit_arr


    return nemodel




##############################################################################
##############################################################################
##############################################################################


def Tmodel_func(c, rs, normsersic, ne_data, tspec_data, nemodel):
    
    '''
    Calculates the non-parameteric model fit to the observed temperature profile. Model T(r) is calculated from Gastaldello+07 Eq. 2, which follows from solving the equation of hydrostatic equilibrium for temperature. 


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
    tfit_arr (array) [keV]: model temperature profile values. Position of model temperature profile is the same as the input tspec_data['radius']

    References:
    -----------
    Gastaldello, F., Buote, D. A., Humphrey, P. J., et al. 2007, ApJ, 669, 158 
    '''

    #return Tmodel given param vals

    ne_ref=nemodel['nefit'][params.refindex]
    tspec_ref=tspec_data['tspec'][params.refindex]
            	
    radius_ref=tspec_data['radius'][params.refindex] 

    tfit_arr=[] #to hold array of fit temperature
    #iterate over all radii values in profile
    for rr in range(0,len(tspec_data)):

        if rr==params.refindex:
            tfit_arr.append(tspec_data['tspec'][rr])
            continue
             
                    
        radius_selected=tspec_data['radius'][rr]
        ne_selected=nemodel['nefit'][rr]



        intfunc = lambda x: intmodel(nemodel['parvals'],rs=rs, c=c, normsersic=normsersic, r_arr=x)

        finfac_t=((params.mu*uconv.mA*uconv.G)/(ne_selected*(uconv.cm_m**-3.))) #[m6 kg-1 s-2] 


        tfit_r=(tspec_ref*ne_ref/ne_selected)-(uconv.joule_kev*finfac_t*(uconv.Msun_kg**2.)*(uconv.kpc_m**-4.)*scipy.integrate.quad(intfunc,radius_ref,radius_selected)[0]) #[kev]

        #print scipy.integrate.quad(intfunc,radius_ref,radius_selected)

        tfit_arr.append(tfit_r)

    return tfit_arr



def calc_rdelta_p(row, nemodel_bfp):


    '''
    Radius corresponding to the input overdensity, i.e. M(Rdelta)/ Vol(Rdelta) = overdensity * rho_crit

    The total mass, DM mass, stellar mass of BCG, and ICM gas mass is then computed within this radius (rdelta).



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

    
    c=row[0]
    rs=row[1]
    normsersic=row[2]


    #rdelta(dm only first)
    rdelta_dm=c*rs
    #this is the radius where the density of dm interior is 500*params.rho_crit
    #within this radius the total mass density will be >500*params.rho_crit, so need a larger radius to get to 500
  
    #calculate mass density at rdelta_dm
    mass_nfw=nfw_mass_model(rdelta_dm,c,rs) #[kg]

    mass_dev=sersic_mass_model(rdelta_dm,normsersic)*uconv.Msun #[kg]


    intfunc = lambda x: mgas_intmodel(rdelta_dm,nemodel_bfp)
    mass_gas=scipy.integrate.quad(intfunc,0,rdelta_dm)[0]*uconv.Msun #[kg]

    
    mass_tot=mass_nfw+mass_dev+mass_gas

    ratio=(mass_tot/((4./3.)*np.pi*rdelta_dm**3.))/params.rho_crit


    #now let's step forward to find true rdelta(total mass)
    rdelta_tot=int(rdelta_dm)
    while ratio>params.overdensity:

        rdelta_tot+=1

        mass_nfw=nfw_mass_model(rdelta_tot,c,rs) #[kg]

        mass_dev=sersic_mass_model(rdelta_tot,normsersic)*uconv.Msun #[kg]

        intfunc = lambda x: mgas_intmodel(rdelta_tot,nemodel_bfp)
        mass_gas=scipy.integrate.quad(intfunc,0,rdelta_tot)[0]*uconv.Msun #[kg]

        mass_tot=mass_nfw+mass_dev+mass_gas

        ratio=(mass_tot/((4./3.)*np.pi*rdelta_tot**3.))/params.rho_crit

    return rdelta_tot, mass_tot/uconv.Msun, mass_nfw/uconv.Msun, mass_dev/uconv.Msun, mass_gas/uconv.Msun






def posterior_mcmc(samples,nemodel):

    '''
    Calculate the radius corresponding to the given overdensity i.e. the radius corresponding to a mean overdensity that is some factor times the critical densiy at the redshift of the cluster. Within this radius, calculate the total mass, DM mass, stellar mass, gas mass.

  
    Args:
    -----
    samples (array): contains the posterior MCMC distribution
            col 0: c
            col 1: rs
            col 2: normsersic

    nemodel_bfp (array): contains parameters for best fitting model to the gas density profile

 
    Returns:
    --------
    samples_aux (array): contains output quantities based on the posterior MCMC distribution
            col 0: Rdelta
            col 1: Mdelta
            col 2: M(DM, delta)
            col 3: M(stars, delta)
            col 4: M(gas, delta)


    Notes:
    ------
    Utilizes JOBLIB for multi-threading. Number of cores as given in params file.
    JOBLIB: https://pythonhosted.org/joblib/

    '''

    samples_aux=Parallel(n_jobs=params.Ncores)(delayed(calc_rdelta_p)(row,nemodel['parvals']) for row in samples)

    return np.array(samples_aux)


##############################################################################
##############################################################################
##############################################################################

'''
MCMC
'''

#define the log likelihood function, here assumes normal distribution
def lnlike(theta, x, y, yerr,ne_data,tspec_data,nemodel):
    '''
    Log(likelihood function) comparing observed and model temperature profile values.

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
    log of the likelood function. Will be a large value when difference between observed and model fits is low. 

    '''
    

    c, rs, normsersic = theta
    
    model = Tmodel_func(c, rs, normsersic, ne_data, tspec_data, nemodel)

    inv_sigma2 = 1.0/(yerr**2) #CHECK THIS!!!
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    
#log-prior
def lnprior(theta):
    '''
    Ensures that free parameters do not go outside bounds

    Args:
    ----
    theta (array): current values of free-paramters; set by WHICH FUNCTION?
    '''
    c, rs, normsersic = theta
    if params.c_boundmin < c < params.c_boundmax and params.rs_boundmin < rs < params.rs_boundmax and params.normsersic_boundmin < normsersic < params.normsersic_boundmax:
        return 0.0
    return -np.inf

#full-log probability function
def lnprob(theta, x, y, yerr,ne_data,tspec_data,nemodel):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr,ne_data,tspec_data,nemodel)



def fit_ml(ne_data,tspec_data,nemodel):

    '''
    Perform maximum likelikhood parameter estimatation. Results can be used as initial guess for more intensive MCMC parameter search.

    Args:
    -----
    ne_data
    tspec_data
    nemodel

    Returns:
    --------
    ml_results (array): results of Maximum-likelihood parameter estimation
       of the form [c_ml,rs_ml,normsersic_ml] which are the best-fitting resutls of the paramter estimation. 


    '''


    nll = lambda *args: -lnlike(*args)

    result = op.minimize(nll, [params.c_guess, params.rs_guess, params.normsersic_guess], args=(tspec_data['radius'],tspec_data['tspec'],tspec_data['tspec_err'],ne_data,tspec_data,nemodel),bounds=((params.c_boundmin,params.c_boundmax),(params.rs_boundmin,params.rs_boundmax),(params.normsersic_boundmin,params.normsersic_boundmax)))


    c_ml, rs_ml, normsersic_ml = result["x"]
    print 'scipy.optimize results'
    print 'ML: c=',c_ml
    print 'ML: rs=',rs_ml
    print 'ML: normsersic=',normsersic_ml

    return [c_ml, rs_ml, normsersic_ml]
    


def fit_mcmc(ne_data,tspec_data,nemodel,ml_results):
    '''
    Run MCMC on the free parameters of model for total gravitating mass. Utilizes EMCEE. 


    Args:
    -----
    ne_data (astropy table): observed gas density profile  (see Notes)
    tspec_data (astropy table): observed temperature profile  (see Notes)
    nemodel (dictionary): best-fitting model to observed gas denisty profile (see Notes)
    ml_results (array): maximum-likelihood paramter estimation for free params
            of the form [c_ml, rs_ml, normsersic_ml]
    
    Returns:
    --------
    samples (array): MCMC sampler chain (??? - i don't know what this means)
            of the form:
                col 1: c
                col 2: rs
                col 3: normsersic
            NB: length of samples array set by params.Nwalkers * params.Nsamples

    References:
    -----------
    EMCEE

    '''

    #initialize walkers - result comes from ML fit before
    ndim, nwalkers = 3, params.Nwalkers
    pos = [ml_results + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


    
    #sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(tspec_data['radius'],tspec_data['tspec'],tspec_data['tspec_err'],ne_data,tspec_data,nemodel),threads=params.Ncores)
    #WHY ARE THE ARGS THE WAY THEY ARE???

    #run mcmc for 500 steps
    sampler.run_mcmc(pos, params.Nsamples)
    samples = sampler.chain[:, params.Nburnin:, :].reshape((-1, ndim))
    #length of samples = walkers*steps
    
    return samples





##############################################################################
##############################################################################
##############################################################################

'''
for mock data testing
'''

def vikh_tprof(pars,r_arr):

    '''
    Temperature profile from Vikhlinin+06 Eq. 4-6 and Table 3. Used for example purposes.
    
    Args:
    -----
    pars (array): parameters of temperature model
    r_arr (array): radii values at which to calculate values of temperature profile
    
    Returns:
    --------
    T3D: temperature profile for given model params in Vikhlinin+06 and input radius values

    References:
    -----------
    Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691
    '''
    
    T0 = pars[0]
    rt = pars[1]
    a = pars[2]
    b = pars[3]
    c = pars[4]
    Tmin = pars[5]
    rcool = pars[6]
    acool = pars[7]


    x=(r_arr/rcool)**acool
    tcool=(x+(Tmin/T0))/(x+1.)
    
    t=((r_arr/rt)**-a)/((1.+((r_arr/rt)**b))**(c/b))
    
    T3D=T0*tcool*t


    return T3D

def vikh_neprof(pars,r):

    '''
    Electron number density profile from Vikhlinin+06 Eq. 3 and Table 2. Used for example purposes.
    
    Args:
    -----
    pars (array): parameters of temperature model
    r_arr (array): radii values at which to calculate values of temperature profile
    
    Returns:
    --------
    ne(r): electron number density  profile for given model params in Vikhlinin+06 and input radius values

    References:
    -----------
    Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691
    '''

    
    
    rdet=pars[0]
    n0=pars[1]
    rc=pars[2]
    rs=pars[3]
    alpha=pars[4]
    beta=pars[5]
    epsilon=pars[6]
    n02=pars[7]
    rc2=pars[8]
    beta2=pars[9]

    gamma=3.


    ne=np.sqrt(params.ne_over_np*(((n0**2.)*((r/rc)**-alpha)/(((1.+(r/rc)**2.)**((3.*beta)-(alpha/2.)))*((1.+(r/rs)**gamma)**(epsilon/gamma))))+((n02**2.)/((1.+(r/rc2)**2.)**(3.*beta2)))))

    return ne





#Features to add: function to determine best fitting density model


#############################################################################
#############################################################################
############################################################################
