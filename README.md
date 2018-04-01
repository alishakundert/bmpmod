# bmpmod

**A backwards mass profile modeler (bmpmod) for galaxy clusters.**

bmpmod is a python package designed to determine the total mass profile of a galaxy cluster via a MCMC fit of input radial profiles of the intracluster medium gas density and temperature.

***  
**Caveat**: This code is undergoing daily development - check back frequently for updates!  
An official release of this package is anticipated for June 2018.
***  

To briefly summarize the analysis process of this package: 

Backwards fitting mass modelling assumes hydrostatic equilibrium and fits the observed temperature profile with a model dependent on a parametric model fit to the gas density profile and an assumed form of the total gravitating mass profile of the cluster. The results of this fit to the temperature profile are the free parameters of the cluster mass model. 

The cluster mass model represents the sum of all mass contributions: Mtot(r) = MDM(r) + Mgas(r) + Mstar(r). In bmpmod, this is accounted for by taking MDM(r) to be the form of a NFW profile with free-params of the concentration parameter and scale radius, Mgas(r) to follow from the gas mass density profile, and optionally Mstar(r) as the contribution of the stellar mass of the central galaxy as modeled by a Sersic profile. 

Full details of this mass profile modeling method as implemented here in bmpmod will be available soon in Kundert et al. 2018, in prep.

To see an example of this code, check out [./example.ipynb](./example.ipynb)!



## Installation

```
git clone https://github.com/alishakundert/massmod.git
```

## Requirements

[numpy](https://github.com/numpy/numpy)\
[matplotlib](https://github.com/matplotlib/matplotlib)\
[scipy](https://github.com/scipy/scipy)\
[astropy](https://github.com/astropy/astropy)\
[emcee](https://github.com/dfm/emcee)\
[acor](https://github.com/dfm/acor)\
[corner](https://github.com/dfm/corner.py)\
[joblib](https://github.com/joblib/joblib)\
[sherpa](https://github.com/sherpa/sherpa)


## Getting started

* Have X-ray data? 

    Use a [template](./template.py) to read in your deprojected gas density and temperature profiles.
    *In development*
    
* No X-ray data? 

    Try out an [example](./example.ipynb) to generate mock density and temperature profiles and explore the available tools of this module.
