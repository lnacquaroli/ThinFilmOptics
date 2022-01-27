# Contiene funciones comunes

import numpy



def find_closest(a, x):
    '''
        Returns the index of the value in the 1d-array x closest to the scalar value a.

            find_closest(a, x)
    '''
    i = numpy.where(numpy.min(numpy.abs(a - x)) == numpy.abs(a - x))[0][0]
    return i

