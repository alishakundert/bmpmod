import numpy as np
import astropy
import astropy.table as atpy
from astropy.table import Column

def set_ne(radius,ne,ne_err,radius_lowerbound=None,radius_upperbound=None):

    '''
    Products correct table format for ne profile data to be used by other functions in this package
    '''


    ne_data=atpy.Table()
    ne_data.add_column(Column(np.array(radius),'radius'))
    ne_data.add_column(Column(np.array(ne),'ne'))
    ne_data.add_column(Column(np.array(ne_err),'ne_err'))

    if (radius_lowerbound is None)&(radius_upperbound is None):
        radius_lowerbound=np.zeros(len(radius))
        radius_upperbound=np.zeros(len(radius))

    
    ne_data.add_column(Column(np.abs(np.array(radius_lowerbound)),'radius_lowerbound'))


    ne_data.add_column(Column(np.abs(np.array(radius_upperbound)),'radius_upperbound'))


    return ne_data


def set_tspec(radius,tspec,tspec_err,tspec_lowerbound=None,tspec_upperbound=None,radius_lowerbound=None,radius_upperbound=None):


    '''
    Products correct table format for temperature profile data to be used by other functions in this package
    '''

    tspec_data=atpy.Table()
    tspec_data.add_column(Column(np.array(radius),'radius'))
    tspec_data.add_column(Column(np.array(tspec),'tspec'))
    tspec_data.add_column(Column(np.array(tspec_err),'tspec_err'))


    if (tspec_lowerbound is None)&(tspec_upperbound is None):
        tspec_lowerbound=tspec_err
        tspec_upperbound=tspec_err

    tspec_data.add_column(Column(np.abs(np.array(tspec_lowerbound)),'tspec_lowerbound'))
    tspec_data.add_column(Column(np.abs(np.array(tspec_upperbound)),'tspec_upperbound'))

    if (radius_lowerbound is None)&(radius_upperbound is None):
        radius_lowerbound=np.zeros(len(radius))
        radius_upperbound=np.zeros(len(radius))


    tspec_data.add_column(Column(np.abs(np.array(radius_lowerbound)),'radius_lowerbound'))
    tspec_data.add_column(Column(np.abs(np.array(radius_upperbound)),'radius_upperbound'))
    
    return tspec_data
