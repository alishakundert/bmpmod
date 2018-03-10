import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import astropy
import astropy.table as atpy
from astropy.table import Column
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


import massmod_func as massmod
import massmod_params as params
import massmod_uconv as uconv




if __name__ == '__main__':

    obsID='4964'


    dirpath='/usr/data/castaway/kundert/obs/'+str(obsID)+'/annuli'

    '''
    read in spec profiles
    '''

    dat=atpy.Table.read(str(dirpath)+'/out_xsp/180202_anco_mincstat_xdep_bfp.txt',format='ascii')   
    #dat=atpy.Table.read(str(dirpath)+'/out_xsp/180120_anco_mincstat_xdep_bfp.txt',format='ascii') 
    #dat=atpy.Table.read(str(dirpath)+'/out_xsp/180120_anco_mincstat_proj_bfp.txt',format='ascii')  
    typedat='xprojct'


    pdat=atpy.Table.read('../particle90_rm_proffit.fits',format='fits')  #deprojected profile from proffit
    conv_dat=atpy.Table.read('/usr/data/castaway/kundert/obs/4964/annuli/out_xsp/18020_conv_soft.txt',format='ascii')  #conversion between count rate and emission measure



    rin_arcsec=np.array(dat['rin_arcsec'])
    rout_arcsec=np.array(dat['rout_arcsec'])
    rpos_arcsec=(((rout_arcsec**(3./2.))+(rin_arcsec**(3./2.)))/2.)**(2./3.)

    r_pos=np.array(rpos_arcsec)*params.kpc_arcsec #[kpc]
    rin_kpc=rin_arcsec*params.kpc_arcsec #[kpc]
    rout_kpc=rout_arcsec*params.kpc_arcsec #[kpc]


    #compute errors on position  values       
    xerr_pos_l=np.array(rpos_arcsec-rin_arcsec)*params.kpc_arcsec #[kpc]  
    xerr_pos_u=np.array(rout_arcsec-rpos_arcsec)*params.kpc_arcsec #[kpc]  

    xerr_pos_l[0]=0.999*xerr_pos_l[0] #small change so loglog can be plotted



    ##begin testing
    ##compare to https://arxiv.org/abs/1403.6416, still can't get correct mnfw
    #z=0.0881
    #c=4.22
    #rdelt=1348.
    #rs=2060/c
    #
    #Hz=H0*np.sqrt(OmegaL+(OmegaM*((1.+z)**3.)))
    #rho_crit=(3.*((Hz*km_Mpc)**2.))/(8.*np.pi*(G*(m_kpc**3.)))  #[kg kpc^-3], should be rho_crit at redshift of cluster
    ##print rho_crit
    #
    #mass_nfw=(10**-14.)*nfw_mass_model(c,rs,rdelt)/Msun
    #
    #print mass_nfw
    #
    #exit()
    ##end testing




 
    '''
    gas density profile
    '''

    norm=np.array(dat['src_norm'])
    norm_el=np.array(-dat['src_norm_el'])
    norm_eu=np.array(dat['src_norm_eu'])


    cosmo=FlatLambdaCDM(H0=70. * u.km/u.s/u.Mpc,Om0=0.3)
    da_mpc=cosmo.angular_diameter_distance(params.z) /u.Mpc
    da_cm=da_mpc*(3.085677581e+24) #angular diameter distance [cm]


    dv=[]
    for ii in range(0,len(dat)):
        dv.append((rout_kpc[ii]**3.)-(rin_kpc[ii]**3.))
    dv=(4./3.)*np.pi*np.array(dv) #[kpc^3]
    dv_cm=dv*((3.08567758128E+21)**3.) #[cm^3]
    #THIS DOESN'T TAKE INTO ACCOUNT PARTIAL ANNULI!!!! - is below ok??? - also need to account for [-2] 
    dv_cm[-1]=dv_cm[-1]*(77.-17.)/360.
    #what was the reference for doing this?


    '''
    spec ne
    '''


    ne_spec=np.sqrt(norm*4.*np.pi*((da_cm*(1.+params.z))**2.)*(10**14.)*(params.ne_over_np/dv_cm))
    ne_spec_el=np.sqrt(norm_el*4.*np.pi*((da_cm*(1.+params.z))**2.)*(10**14.)*(params.ne_over_np/dv_cm))
    ne_spec_eu=np.sqrt(norm_eu*4.*np.pi*((da_cm*(1.+params.z))**2.)*(10**14.)*(params.ne_over_np/dv_cm))
    #ne_err=np.sqrt((ne_el**2.)+(ne_eu**2.)) HOW ARE ERRORS COMBINED?
    ne_spec_err=(np.abs(ne_spec_el)+np.abs(ne_spec_eu))/2.


    '''
    sb ne
    '''
    pdat['RADIUS']=np.array(pdat['RADIUS'])*60. #[arcsec]
    pdat['WIDTH']=np.array(pdat['WIDTH'])*60. #[arcsec]

        
    pdat.add_column(astropy.table.Column(np.zeros(len(pdat)),name='conv'))
    for ii in range(0,len(conv_dat)):
        conv=conv_dat['conv'][ii]
        rin_arcsec=conv_dat['rin_arcsec'][ii]
        rout_arcsec=conv_dat['rout_arcsec'][ii]

        binind=np.where((np.array(pdat['RADIUS'])>=rin_arcsec)&(np.array(pdat['RADIUS'])<rout_arcsec))[0]
        pdat['conv'][binind]=conv


    ne_sb=np.sqrt(((np.array(pdat['DEPR']))/pdat['conv'])*params.ne_over_np*4.*np.pi*((da_cm*(1.+params.z))**2.)*(10**14.)*(uconv.kpc_cm**-3.))
    ne_sb_err=0.5*np.array(pdat['ERR_DEPR'])*(np.array(pdat['DEPR'])**-0.5)*np.sqrt((1./pdat['conv'])*params.ne_over_np*4.*np.pi*((da_cm*(1.+params.z))**2.)*(10**14.)*(uconv.kpc_cm**-3.)) #simple uncertaintity propagation > write out if you get confused later

    pdat.add_column(astropy.table.Column(ne_sb_err,name='ERR_NE'))

    pdat['RADIUS']=np.array(pdat['RADIUS'])*params.kpc_arcsec #[kpc]
    pdat['WIDTH']=np.array(pdat['WIDTH'])*params.kpc_arcsec #[kpc]

    #pdat.add_column(astropy.table.Column(ne_sb,name='NE'))
    #pdat.add_column(astropy.table.Column(ne_sb_err,name='ERR_NE'))

    #this recalcualtes the ne density based on the updated conversion rates
    pdat['DENSITY']=ne_sb
    pdat['ERR_DENS']=ne_sb_err

    #remove nan values from pdat
    ind=np.where(np.isnan(pdat['DENSITY'])==False)[0]
    pdat=pdat[ind]
    

    #add final point for spec density profile
    pdat.add_row()
    pdat['RADIUS'][-1]=r_pos[-1]
    pdat['WIDTH'][-1]=(rout_kpc[-1]-rin_kpc[-1])/2.
    pdat['DENSITY'][-1]=ne_spec[-1]
    pdat['ERR_DENS'][-1]=ne_spec_err[-1]


    #initialize temperature data
    tspec_arr=np.array(dat['src_kT']) #measured array of spec temperature
    tspec_el=np.array(dat['src_kT_el'])
    tspec_eu=np.array(dat['src_kT_eu'])
    #tspec_err=np.sqrt((tspec_el**2.)+(tspec_eu**2.))
    tspec_err=(np.abs(tspec_el)+np.abs(tspec_eu))/2.



    '''
    set up data arrays
    '''
    ne_data=atpy.Table()
    ne_data.add_column(Column(pdat['RADIUS'],'radius'))
    ne_data.add_column(Column(pdat['DENSITY'],'ne'))
    ne_data.add_column(Column(pdat['ERR_DENS'],'ne_err'))
    ne_data.add_column(Column(pdat['WIDTH'],'radius_lowerbound'))
    ne_data.add_column(Column(pdat['WIDTH'],'radius_upperbound'))

    tspec_data=atpy.Table()
    tspec_data.add_column(Column(r_pos,'radius'))
    tspec_data.add_column(Column(dat['src_kT'],'tspec'))
    tspec_data.add_column(Column(tspec_err,'tspec_err'))
    tspec_data.add_column(Column(-tspec_el,'tspec_lowerbound'))
    tspec_data.add_column(Column(tspec_eu,'tspec_upperbound'))
    tspec_data.add_column(Column(xerr_pos_l,'radius_lowerbound'))
    tspec_data.add_column(Column(xerr_pos_u,'radius_upperbound'))
    


    '''
    fit density profile with beta model - py sherpa
    '''
    
    #need to generalize this a lot to remove double betamodel 
    nemodel=massmod.fitne(ne_data,tspec_data) #[cm^-3]


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

    #plotting ne spectral results - just for my data, don't use for examples
    ax=fig2.add_subplot(2,2,1) 
    plt.loglog(r_pos, ne_spec,'bo')
    plt.errorbar(r_pos,ne_spec,xerr=[xerr_pos_l,xerr_pos_u],yerr=[ne_spec_el,ne_spec_eu],linestyle='none',color='b')



    ##########################################################################
    ######################################################################### 
    ##########################################################################


    plt.tight_layout()
    plt.show()

    print '/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'.pdf'
    #fig1.savefig('/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'_mcmc.pdf',dpi=300,format='PDF',bbox_inches='tight')
    #fig2.savefig('/usr/data/castaway/kundert/obs/'+str(obsID)+'/outplot/'+str(obsID)+'_massmod_ref'+str(params.refindex)+'.pdf',dpi=300,format='PDF',bbox_inches='tight')



'''
Deprecated
'''

#    if params.fitpdat==0: #using apec norm gas densities
#        if params.fitbetamodel == 1:
#            nemodel_bfp=massmod.fitne_betamodel(r_pos,ne,ne_err)
#            nemodel_bfp[0]=nemodel_bfp[0]*params.mu_e*uconv.mA #to change ne>rho
#            rhofit_arr=massmod.betamodel(nemodel_bfp,r_pos) #[kg cm-3]
#        elif params.fitbetamodel == 2:
#            nemodel_bfp=massmod.fitne_doublebetamodel(r_pos,ne,ne_err)
#            nemodel_bfp[0]=nemodel_bfp[0]*params.mu_e*uconv.mA #to change ne>rho
#            nemodel_bfp[3]=nemodel_bfp[3]*params.mu_e*uconv.mA #to change ne>rho
#            rhofit_arr=massmod.doublebetamodel(nemodel_bfp,r_pos) #[kg cm-3]
#        elif params.fitbetamodel == 0:
#            nemodel_bfp=massmod.fitne_cuspedbetamodel(r_pos,ne,ne_err)
#            nemodel_bfp[0]=nemodel_bfp[0]*params.mu_e*uconv.mA #to change ne>rho
#            rhofit_arr=massmod.cuspedbetamodel(nemodel_bfp,r_pos) #[kg cm-3]
#
#    elif params.fitpdat==1: #using proffit deprojected data
#        if params.fitbetamodel == 1:
#            nemodel_bfp=massmod.fitne_betamodel(pdat['RADIUS'],pdat['DENSITY'],pdat['ERR_DENS'])
#            nemodel_bfp[0]=nemodel_bfp[0]*params.mu_e*uconv.mA #to change ne>rho
#            rhofit_arr=massmod.betamodel(nemodel_bfp,r_pos) #[kg cm-3]
#
#
#        elif params.fitbetamodel == 2:
#            nemodel_bfp=massmod.fitne_doublebetamodel(pdat['RADIUS'],pdat['DENSITY'],pdat['ERR_DENS'])
#            nemodel_bfp[0]=nemodel_bfp[0]*params.mu_e*uconv.mA #to change ne>rho
#            nemodel_bfp[3]=nemodel_bfp[3]*params.mu_e*uconv.mA #to change ne>rho
#            rhofit_arr=massmod.doublebetamodel(nemodel_bfp,r_pos) #[kg cm-3]
#
#
#        elif params.fitbetamodel == 0:
#            nemodel_bfp=massmod.fitne_cuspedbetamodel(pdat['RADIUS'],pdat['DENSITY'],pdat['ERR_DENS'])
#            nemodel_bfp[0]=nemodel_bfp[0]*params.mu_e*uconv.mA #to change ne>rho
#            rhofit_arr=massmod.cuspedbetamodel(nemodel_bfp,r_pos) #[kg cm-3]
#
