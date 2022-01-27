# This file aims to contain the effective refractive index models
# The first attempt is to use functions only

import numpy


def lorentz_lorenz(n_1, n_2, f):
    '''
        Effective index of refraction of a binary mix using the Lorentz-Lorenz model.

            neff = lorentz_lorenz(n_1, n_2, f)

        n_1: is the index of refraction of the component 1
        n_2: is the index of refraction of the component 2
        f: is the fraction of component 1
        
        neff: Complex effective index of refraction
    '''
    n1_sq, n2_sq = complex(n_1**2), complex(n_2**2)
    aux = f*(n1_sq - 1.0)/(n1_sq  + 2.0) + (1.0 - f)*(n2_sq - 1.0)/(n2_sq + 2.0)
    neff = numpy.sqrt(-1.0 - 2.0*aux)/numpy.sqrt(aux - 1.0)
    return neff


def maxwell_garnett(n_1, n_2, f):
    '''
        Effective index of refraction of a binary mix using the Maxwell-Garnett model.

            neff = maxwell_garnett(n_1, n_2, f)

        n_1: is the index of refraction of the component 1
        n_2: is the index of refraction of the component 2
        f: is the fraction of component 1

        neff: Complex effective index of refraction
    '''
    e_1, e_2 = complex(n_2**2), complex(n_1**2) # flipped so f belongs to n_1
    e2_times_e1 = e_2*e_1
    neff = numpy.sqrt((-3.0*e2_times_e1 + 2.0*f*(e2_times_e1 - e_2**2)) \
            / (-3.0*e_2 + f*(e_2 - e_1)))
    return neff


def bruggeman(n_1, n_2, f):
    '''
        Effective index of refraction of a binary mix using the Bruggeman model.

            neff = bruggeman(n_1, n_2, f)

        n_1: is the index of refraction of the component 1
        n_2: is the index of refraction of the component 2
        f: is the fraction of component 1

        neff: Complex effective index of refraction
    '''
    e_1, e_2 =  complex(n_2**2), complex(n_1**2) # flipped so f belongs to n_1
    e1_times_2 = 2.0*e_1
    A = 3.0*f*(e_2 - e_1)
    neff = numpy.sqrt((e1_times_2 - e_2 + A \
            + numpy.sqrt(8.0*e_1*e_2 \
                + (e1_times_2 - e_2 + A)**2))/4.0)
    return neff


def looyenga(n_1, n_2, f):
    '''
        Effective index of refraction of a binary mix using the Looyenga model.
            neff = looyenga(n_1, n_2, f)

        n_1: is the index of refraction of the component 1
        n_2: is the index of refraction of the component 2
        f: is the fraction of component 1

        neff: Complex effective index of refraction

        Ref: IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 38, NO. 3, MAY 2000
    '''
    neff = (((1.0 - f)*(n_2**(2/3))) + ((n_1**(2/3))*f))**(3/2)
    return neff


def inverse_looyenga(x):
    '''
        Returns the physical thickness for a given porosity and optical thickness using the Looyenga model with two components.

            inverse_looyenga(x)
        
        x: array with porosity and optical thickness
    '''
    d = x[1]/(1.6*(1.0 - x[0]) + 1.0)**(1.5)
    return d