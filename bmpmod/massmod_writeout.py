import numpy as np


def write_ne(nemodel, fn):

    for ii in range(0, len(nemodel['parvals'])):
        if ii == 0:  # ne01
            print ii, '$'+str(np.round(nemodel['parvals'][ii]*(10**1.), 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii]*(10**1.), 2)) \
                + '}^{'+str(np.round(nemodel['parmaxes'][ii]*(10**1.), 2))+'}$'
            continue

        if ii == 1:  # rc1
            print ii, '$'+str(np.round(nemodel['parvals'][ii], 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii], 2)) \
                + '}^{'+str(np.round(nemodel['parmaxes'][ii], 2))+'}$'
            continue

        if ii == 2:  # beta1
            print ii, '$'+str(np.round(nemodel['parvals'][ii], 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii], 2)) \
                + '}^{'+str(np.round(nemodel['parmaxes'][ii], 2))+'}$'
            continue

        if ii == 3:  # ne02
            print ii, '$'+str(np.round(nemodel['parvals'][ii]*(10**3.), 2)) \
                + '_{'+str(np.round(nemodel['parmins'][ii]*(10**3.), 2)) \
                + '}^{'+str(np.round(nemodel['parmaxes'][ii]*(10**3.), 2))+'}$'
            continue

        if ii == 4:  # rc2
            print ii, '$'+str(int(np.round(nemodel['parvals'][ii], 0))) \
                + '_{'+str(int(np.round(nemodel['parmins'][ii], 0))) \
                + '}^{'+str(int(np.round(nemodel['parmaxes'][ii], 0)))+'}$'
            continue
