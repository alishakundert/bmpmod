# massmod

A python package to perform a mass-modelling analysis of ICM gas and temperature profiles obtained from X-ray observations.

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

* No X-ray data? 

    Try out an [example](./example.py) to generate mock density and temperature profiles and explore the available tools of this module.
