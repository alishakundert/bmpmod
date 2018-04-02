import numpy as np
import astropy.table as atpy
from astropy.table import Column


def set_ne(radius,
           ne,
           ne_err,
           radius_lowerbound=None,
           radius_upperbound=None):

    '''
    Establishes correct table format for gas density profile data to be used by
    other functions in this package.

    Args:
    -----
    radius [kpc] (array): central radial bin values of ne profile
    ne [cm^-3] (array): electron number density values of profile
    ne_err [cm^-3] (array): error bars on ne values
    radius_lowerbound [kpc] (array): lower values of radial bins
    radius_upperbound [kpc] (array): upper values of radial bins

    Returns:
    --------
    ne_data (astropy table): table containing gas density profile information
        with the required format for bmpmod
    '''

    ne_data = atpy.Table()
    ne_data.add_column(Column(np.array(radius), 'radius'))
    ne_data.add_column(Column(np.array(ne), 'ne'))
    ne_data.add_column(Column(np.array(ne_err), 'ne_err'))

    if (radius_lowerbound is None) & (radius_upperbound is None):
        radius_lowerbound = np.zeros(len(radius))
        radius_upperbound = np.zeros(len(radius))

    ne_data.add_column(Column(np.abs(np.array(radius_lowerbound)),
                              'radius_lowerbound'))

    ne_data.add_column(Column(np.abs(np.array(radius_upperbound)),
                              'radius_upperbound'))

    return ne_data


def set_tspec(radius,
              tspec,
              tspec_err,
              tspec_lowerbound=None,
              tspec_upperbound=None,
              radius_lowerbound=None,
              radius_upperbound=None):

    '''
    Establishes correct table format for temperature profile data to be used by
    other functions in this package.

    Args:
    -----
    radius [kpc] (array): central radial bin values of temperature profile
    tspec [kev] (array): profile temperature values
    tspec_err [keV] (array): error bars on temperature values
    tspec_lowerbound [keV] (array): lower error bound on temperature values
    tspec_upperbound [keV] (array): upper error bound on temperature values
    radius_lowerbound [kpc] (array): lower values of radial bins
    radius_upperbound [kpc] (array): upper values of radial bins

    Returns:
    --------
    tspec_data (astropy table): table containing temperature profile information
        with the required format for bmpmod
    '''

    tspec_data = atpy.Table()
    tspec_data.add_column(Column(np.array(radius), 'radius'))
    tspec_data.add_column(Column(np.array(tspec), 'tspec'))
    tspec_data.add_column(Column(np.array(tspec_err), 'tspec_err'))

    if (tspec_lowerbound is None) & (tspec_upperbound is None):
        tspec_lowerbound = tspec_err
        tspec_upperbound = tspec_err

    tspec_data.add_column(Column(np.abs(np.array(tspec_lowerbound)),
                                 'tspec_lowerbound'))
    tspec_data.add_column(Column(np.abs(np.array(tspec_upperbound)),
                                 'tspec_upperbound'))

    if (radius_lowerbound is None) & (radius_upperbound is None):
        radius_lowerbound = np.zeros(len(radius))
        radius_upperbound = np.zeros(len(radius))

    tspec_data.add_column(Column(np.abs(np.array(radius_lowerbound)),
                                 'radius_lowerbound'))
    tspec_data.add_column(Column(np.abs(np.array(radius_upperbound)),
                                 'radius_upperbound'))

    return tspec_data


def set_meta(name,
             z,
             bcg_re=0,
             bcg_sersic_n=0,
             refindex=-1,
             incl_mstar=0,
             incl_mgas=0):

    '''
    Creates dictionary with important information about the cluster and
    analysis options.

    Args:
    -----
    name (string): name of cluster
    z (float): redshift of cluster
    bcg_re (float): effective radius of cluster central galaxy
    bcg_sersic_n (float): Sersic index of cluster central galaxy
    refindex (int): index into the temperature profile where Tmodel= Tspec
    incl_mstar (int): option to include the stellar mass of the central galaxy
        into the total gravitating mass model
    incl_mgas (int): option to include the mass of the ICM into the total
        gravitating mass model

    Returns:
    --------
    clustermeta (dictionary): dictionary of cluster and analysis info stored
        into the proper format to be used by bmpmod functions
    '''

    clustermeta = {}
    clustermeta['name'] = name
    clustermeta['z'] = z
    clustermeta['refindex'] = refindex
    clustermeta['incl_mstar'] = incl_mstar
    clustermeta['incl_mgas'] = incl_mgas

    clustermeta['bcg_re'] = bcg_re
    clustermeta['bcg_sersic_n'] = bcg_sersic_n

    if (incl_mstar == 1) & ((bcg_re == 0) | (bcg_sersic_n == 0)):
        print 'Re and n of BCG required to count mstar contribution'
        exit()

    return clustermeta
