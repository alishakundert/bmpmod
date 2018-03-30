# bmpmod

bmpmod is a python package designed to fit for the total mass profile of a galaxy cluster given input radial profiles of the intracluster medium gas density and temperature. 

***  
**Caveat**: This code is undergoing daily development - check back frequently for updates! 
An official release of this package is anticipated for June 2018.
***  

To briefly summarize the analysis process of this package, backwards fitting mass modelling assumes hydrostatic equilibrium and fits the temperature profile with a model of the form:

$kT(R) = \frac{kT(R_{\mathrm{ref}}) \ n_{e}(R_{\mathrm{ref}})}{n_{e}(R)} -\frac{\mu m_{p} G}{n_{e}(R)}
\int_{R_{\mathrm{ref}}}^R \frac{n_{e}(r) M_{\mathrm{grav}}(r)}{r^2} dr$  


Where \rho_gas(r) is the parametric model fit to the gas density profile, and Mtot is the total gravitating mass model of the form Mtot = MDM+Mgas+Mstars.

The end goal of this process is to determine the free parameters of Mtot, which is accomplished in the fit of the temperature model to the temperature profile via a MCMC analysis. This package takes MDM to be the form of a NFW profile with free-params of the concentration parameter and scale radius, Mgas to follow from the gas mass density profile, and optionally Mstar as the contribution of the stellar mass of the central galaxy modeled by a Sersic profile. To see an example of this code in action, check out an [example](./example.ipynb)!

Full details of this method will be available soon in Kundert et al. 2018, in prep.


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

    Use [template](./template.py) to read in your deprojected gas density and temperature profiles.
    *In development*
    
* No X-ray data? 

    Try out an [example](./example.py) to generate mock density and temperature profiles and explore the available tools of this module.
