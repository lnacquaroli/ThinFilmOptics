# Contiene varias funciones para realizar ajustes de espectros de reflectancia de SP.
# Es necesario tener los archivos EffectiveIndexModels.py y RefractiveIndexDB.py en la misma carpeta o path.

import numpy
from scipy.interpolate import interp1d

# Solo para los defaults en las funciones
from EffectiveIndexModels import looyenga
from RefractiveIndexDB import silicon
from CommonUtils import BeamParameters, find_closest


def merito1(
    x,
    ref_experimental,  # vector of spectrum
    n_incident,  # index of refraction incident layer
    n_substrate,  # index of refraction substrate layer
    beam, # parameters of the beam
    n_void, # index of refraction of void inside the effective medium
    n_matrix, # index of refraction of matrix (solid) inside the effective medium
    effective_index_binary_func, # effective medium approximation
    inverse_ema_func, # EMA to recover the physical thickness
):
    '''
        Ajusta espectro de reflectancia de una capa simple con el modelo de Looyenga usando coeficientes de Fresnel.
    '''

    n_effective = effective_index_binary_func(n_void, n_matrix, x[0])

    # calculation of the angle of incidence inside each media with Snell law of refraction
    phi0 = beam.angle_inc_rad
    phi1 = numpy.arcsin(n_incident*numpy.sin(phi0)/n_effective) # bulk
    phi2 = numpy.arcsin(n_effective*numpy.sin(phi1)/n_substrate) # substrate

    # fresnel coefficients averaging the two polarizations
    cosphi0 = numpy.cos(phi0)
    cosphi1 = numpy.cos(phi1)
    cosphi2 = numpy.cos(phi2)
    rp01 = (n_incident*cosphi1 - n_effective*cosphi0) \
            / (n_incident*cosphi1 + n_effective*cosphi0)
    rp12 = (n_effective*cosphi2 - n_substrate*cosphi1) \
            / (n_effective*cosphi2 + n_substrate*cosphi1)
    rs01 = (n_incident*cosphi0 - n_effective*cosphi1) \
            / (n_incident*cosphi0 + n_effective*cosphi1)
    rs12 = (n_effective*cosphi1 - n_substrate*cosphi2) \
            / (n_effective*cosphi1 + n_substrate*cosphi2)
    
    # phase shift
    d = inverse_ema_func(x)
    beta = 12.566370614359172*d*n_effective*numpy.cos(phi1)/beam.wavelength
    # 4*pi = 12.566370614359172

    # complex reflectance
    exp_beta = numpy.exp(1j*beta)
    Ap, As = rp12*exp_beta, rs12*exp_beta
    rp012 = (rp01 + Ap)/(1.0 + rp01*Ap)
    rs012 = (rs01 + As)/(1.0 + rs01*As)
    
    # reflectance
    Refp, Refs = (rp012*numpy.conj(rp012)).real, (rs012*numpy.conj(rs012)).real
    reflectance = ((1.0 - beam.polarisation)*Refs) + (beam.polarisation*Refp)
    
    # objective function: mean-abs
    cost = numpy.mean(numpy.abs(reflectance - ref_experimental))

    return cost


def inverse_looyenga(x):
    '''
        Returns the physical thickness for a given porosity and optical thickness using the Looyenga model with two components.

            inverse_looyenga(x)
        
        x: array with porosity and optical thickness
    '''
    d = x[1]/(1.6*(1.0 - x[0]) + 1.0)**(1.5)
    return d


def normalize_experimental_reflectance(r, r_ref, r_theory, beam, kind="cubic"):
    itp_r = interp1d(r[:, 0], r[:, 1], kind=kind)
    itp_ref = interp1d(r_ref[:, 0], r_ref[:, 1], kind=kind)
    r_ = itp_r(beam.wavelength) / itp_ref(beam.wavelength) * r_theory
    return r_


def reflectance_silicon(beam):
    '''
        Returns the reflectance of pure silicon for a given angle of incidence and polarization in a given range of wavelengths.
        
            reflectance_silicon(beam)
    '''
    n = silicon(beam.wavelength)
    R = numpy.zeros(len(n))
    for i, l in enumerate(beam.wavelength):
        N = numpy.stack((1.0+1j*0, n[i], n[i]))
        R[i] = _reflectance_spectrum(N, [100], l, beam.angle_inc_rad, beam.polarisation)
    return R


def reflectance(n_layers, d_vec, beam):
    '''
        Returns the reflectance of a n_layers structure for a given beam structure.
        
            reflectance(n_layers, d_vec, beam)

        n_layers: matrix with the structure of indices of refraction of each layer per column that compose the system
        d_vec: array with the thicknesses of each layer
        beam: structure with the beam parameters
    '''
    R = numpy.zeros(len(beam.wavelength))
    for i, l in enumerate(beam.wavelength):
        R[i] = _reflectance_spectrum(
            n_layers[i, :], d_vec, l, beam.angle_inc_rad, beam.polarisation,
        )
    return R


def _reflectance_spectrum(n_layers, d_vec, wavelength, phi, w):
    '''
        Returns the reflection spectrum using the matrix method
            
            _refletance_spectrum(n_layers, d_vec, wavelength, phi, w)
    '''    
    # number of layers in the structure
    numlay = len(n_layers) - 1
    # initialize variables
    adm_s = numpy.zeros(numlay + 1, dtype=complex)
    adm_p, phi_ = adm_s.copy(), adm_s.copy()
    delta = numpy.zeros(numlay + 1, dtype=complex)
    Ms, Mp = numpy.eye(2, dtype=complex), numpy.eye(2, dtype=complex)
    # Calculate the admittance of the medium
    phi_[0] = phi #numpy.copy(phi)
    adm_s[0], adm_p[0] = _admittance(n_layers[0], phi_[0])
    # calculation of the optical transfer matrix for each layer between the
    # incident medium and the substrate
    for c in range(1, numlay):
        # compute angles inside each layer according to the Snell law
        phi_[c] = numpy.arcsin(n_layers[c - 1]/n_layers[c]*numpy.sin(phi_[c - 1]))
        # phase shifts for each layer: 2*pi = 6.283185307179586
        delta[c] = 6.283185307179586*n_layers[c]*d_vec[c - 1]*numpy.cos(phi_[c])/wavelength
        adm_s[c], adm_p[c] = _admittance(n_layers[c], phi_[c])
        # total transfer matrix
        Ms = numpy.matmul(Ms, _tmatrix(delta[c], adm_s[c]))
        Mp = numpy.matmul(Mp, _tmatrix(delta[c], adm_p[c]))
    # compute the admittance of the substrate
    phi_[numlay] = numpy.arcsin(n_layers[numlay - 1]/n_layers[numlay]*numpy.sin(phi_[numlay - 1]))
    adm_s[numlay], adm_p[numlay] = _admittance(n_layers[numlay], phi_[numlay])
    # calculation of the equivalent admittance of the entire multilayer
    b, c, d = Ms[1,0]/adm_s[0], Ms[0,1]*adm_s[numlay], Ms[1,1]*adm_s[numlay]/adm_s[0]
    A, C = Ms[0,0] - b + c - d, Ms[0,0] + b + c + d
    b, c, d = Mp[1,0]/adm_p[0], Mp[0,1]*adm_p[numlay], Mp[1,1]*adm_p[numlay]/adm_p[0]
    B, D = Mp[0,0] - b + c - d, Mp[0,0] + b + c + d
    r = ((1.0 - w)*A + w*B)/((1.0 - w)*C + w*D)
    R = r*numpy.conj(r)
    return R.real


def _admittance(n, phi):
    cosphi = numpy.cos(phi)
    eta_p, eta_s = n/cosphi, n*cosphi
    return eta_s, eta_p


def _tmatrix(beta, adm):
    # Optic transfer matrix (iternal function)
    cos_beta, sin_beta = numpy.cos(beta), -1j*numpy.sin(beta)
    Q = numpy.array([
        [cos_beta, sin_beta/adm],
        [adm*sin_beta, cos_beta],
    ])
    return Q

    
def solution_space_1(
    FLAG_1, # True or False
    LB, UB,
    beam,
    ref_experimental,
    n_incident,
    n_substrate,
    n_void,
    n_matrix,
    NUM_GRID=40,
    effective_index_binary_func=looyenga,
    inverse_ema_func=inverse_looyenga, # EMA to recover the physical thickness
):
    '''
        Calcula el espacio de soluciones para Looyenga, con mean_abs como funcion objetivo.
        Devuelve los valores aux1 (porosidades), aux2 (espesores opticos), S (funcion objetivo) y s que contiene los valores de porosidad y espesor para el minimo de S.

        El espacio de soluciones se puede graficar usando aux1, aux2 y S.
    '''
    # Calcula el espacio de soluciones
    if FLAG_1:
        aux1 = numpy.linspace(LB[0], UB[0], NUM_GRID) 
        aux2 = numpy.linspace(LB[1], UB[1], NUM_GRID) 
        error_surface = numpy.zeros((len(aux1), len(aux2)))
        for j in range(error_surface.shape[0]):
            for k in range(error_surface.shape[1]):
                error_surface[j, k] = merito1(
                    [aux1[j], aux2[k]],
                    ref_experimental,
                    n_incident, 
                    n_substrate,
                    beam,
                    n_void,
                    n_matrix,
                    effective_index_binary_func,
                    inverse_ema_func,
                )
        i1, i2 = numpy.where(error_surface == numpy.min(error_surface))
        s = [aux1[i1][0], aux2[i2][0]]
    else:
        s = numpy.nan
        error_surface = numpy.nan
        aux1, aux2 = numpy.nan, numpy.nan
    return error_surface, numpy.min(error_surface), s, aux1, aux2


def merito6(
    x,
    ref_experimental,
    n_incident,
    n_substrate,
    beam,
    n_void,
    n_matrix, 
    NUM_LAYERS=100,
    effective_index_binary_func=looyenga,
):
    # Funcion que ajusta un espectro de reflectancia de una capa simple que posee un gradiente de porosidad debido a la disolucion de la capa porosa durante el proceso de anodizado
    # Utiliza la regla de mezcla de looyenga y el metodo de matrices para el calculo. 
    # En esta funcion se supone una variacion de porosidad con la profundidad de la multicapa mediante el parametro alfa.
    pvec, dvec = linear_porosity(x, NUM_LAYERS)
    R = numpy.zeros(len(beam.wavelength))
    for i, l in enumerate(beam.wavelength):
        N = numpy.concatenate([
            n_incident[i],
            effective_index_binary_func(n_void[i], n_matrix[i], pvec),
            n_substrate[i],
            ])
        R[i] = _reflectance_spectrum(N, dvec, l, beam.ang_inc_rad, beam.polarisation)
    
    cost = numpy.mean(numpy.abs(R - ref_experimental))
    return cost


def linear_porosity(x, NUM_LAYERS):
    ''' 
    Build the linear porosity variation in terms of the thickness
    '''
    pvec = x[0] + x[2]*(1.0 - numpy.linspace(1, NUM_LAYERS, num=NUM_LAYERS)/NUM_LAYERS)
    dvec = numpy.ones(NUM_LAYERS)*x[1]/NUM_LAYERS
    return pvec, dvec


def merito2(
    x,
    ref_experimental,
    n_incident, 
    n_substrate,
    beam,
    n_void, 
    n_matrix,
    effective_index_binary_func=looyenga,
    NUM_BRAGGS=4,
    NUM_DEFECT=2,
):
    '''
        Funcion que ajusta un espectro de reflectancia de una multicapa porosa compuesta alternando dos capas distintas.
        Utiliza la regla de mezcla indicada y el metodo de matrices para el calculo. 
        En esta funcion se supone una variacion de espesores con la profundidad de la multicapa mediante el parametro alfa (no confundir con disolucion).
    '''
    p, d = numpy.array(x[0:2]), numpy.array(x[2:5])
    alfa = x[3]
    vec1 = numpy.arange(2*NUM_BRAGGS + NUM_DEFECT)
    _aux = numpy.tile(p, NUM_BRAGGS)
    pvec = numpy.concatenate(
        (_aux, numpy.tile(p[0], NUM_DEFECT), numpy.flip(_aux))
    )
    _aux = numpy.tile(d, NUM_BRAGGS)
    dvec = numpy.concatenate(
        (_aux, numpy.tile(d[0], NUM_DEFECT), numpy.flip(_aux))
    )*numpy.concatenate(([1], alfa**vec1))
    R = numpy.zeros(len(beam.wavelength))
    for i, l in enumerate(beam.wavelength):
        N = numpy.concatenate([
            n_incident[i],
            effective_index_binary_func(n_void[i], n_matrix[i], pvec),
            n_substrate[i],
        ])
        R[i] = _reflectance_spectrum(N, dvec, l, beam.ang_inc_rad, beam.polarisation)
    cost = numpy.mean(numpy.abs(R - ref_experimental))
    return cost


def merito3(
    x,
    ref_experimental,
    n_incident, 
    n_substrate,
    beam,
    n_void, 
    n_matrix,
    effective_index_binary_func=looyenga,
    NUM_BRAGGS=4,
    NUM_DEFECT=2,
):
    '''
        Funcion que ajusta un espectro de reflectancia de una multicapa porosa compuesta alternando dos capas distintas.
        En esta funcion se supone una variacion de espesores con la profundidad de la multicapa mediante el parametro alfa (no confundir con disolucion).
        Aqui se ajusta tambien el espesor del defecto suponiendo que es diferente.
    '''
    p, d = numpy.array(x[0:2]), numpy.array(x[2:5])
    alfa = x[5]
    vec1 = numpy.arange(2*NUM_BRAGGS + NUM_DEFECT)
    _aux = numpy.tile(p, NUM_BRAGGS)
    pvec = numpy.concatenate(
        (_aux, numpy.tile(p[0], NUM_DEFECT), numpy.flip(_aux))
    )
    _aux = numpy.tile(d, NUM_BRAGGS)
    dvec = numpy.concatenate(
        (_aux, numpy.tile(d[2], NUM_DEFECT), numpy.flip(_aux))
    )*numpy.concatenate(([1], alfa**vec1))
    R = numpy.zeros(len(beam.wavelength))
    for i, l in enumerate(beam.wavelength):
        N = numpy.concatenate([
            n_incident[i],
            effective_index_binary_func(n_void[i], n_matrix[i], pvec),
            n_substrate[i],
        ])
        R[i] = _reflectance_spectrum(N, dvec, l, beam.ang_inc_rad, beam.polarisation);
    cost = numpy.mean(numpy.abs(R - ref_experimental))
    return cost


def merito4(
    x,
    ref_experimental,
    n_incident, 
    n_substrate,
    beam,
    n_void, 
    n_matrix,
    effective_index_binary_func=looyenga,
    NUM_BRAGGS=4,
    NUM_DEFECT=2,
    p_1=0.75,
):
    '''
        Funcion que ajusta un espectro de reflectancia de una multicapa porosa compuesta alternando dos capas distintas.
        En esta funcion se supone una variacion de espesores con la profundidad de la multicapa mediante el parametro alfa (no confundir con disolucion).
        Aqui usa una porosidad diferente para la primera capa y un espesor diferente para ajustar la cavidad.
    '''
    p, d = numpy.array(x[0:2]), numpy.array(x[2:5])
    alfa = x[5]
    vec1 = numpy.arange(2*NUM_BRAGGS + NUM_DEFECT)
    _aux = numpy.tile(p, NUM_BRAGGS)
    pvec = numpy.concatenate(
        ([p_1], _aux, numpy.tile(p[0], NUM_DEFECT), numpy.flip(_aux))
    )
    _aux = numpy.tile(d, NUM_BRAGGS)
    dvec = numpy.concatenate(
        (_aux, numpy.tile(d[3], NUM_DEFECT), numpy.flip(_aux))
    )*numpy.concatenate(([1], alfa**vec1))
    R = numpy.zeros(len(beam.wavelength))
    for i, l in enumerate(beam.wavelength):
        N = numpy.concatenate([
            n_incident[i],
            effective_index_binary_func(n_void[i], n_matrix[i], pvec),
            n_substrate[i],
        ])
        R[i] = _reflectance_spectrum(N, dvec, l, beam.ang_inc_rad, beam.polarisation);
    cost = numpy.mean(numpy.abs(R - ref_experimental))
    return cost


def merito5(
    x,
    ref_experimental,
    n_incident, 
    n_substrate,
    beam,
    n_void, 
    n_matrix,
    effective_index_binary_func=looyenga,
    NUM_BRAGGS=4,
    NUM_DEFECT=2,
    d_2=2500,
):
    '''
        Funcion que ajusta un espectro de reflectancia de una multicapa porosa compuesta alternando dos capas distintas.
        En esta funcion se supone una variacion de espesores con la profundidad de la multicapa mediante el parametro alfa (no confundir con disolucion).
        Aqui usa un espesor fijo en la segunda capa y se ajusta el espesor de la cavidad.
    '''
    p, d = numpy.array(x[0:2]), numpy.array(x[2:5])
    alfa = x[5]
    vec1 = numpy.arange(2*NUM_BRAGGS + NUM_DEFECT)
    _aux = numpy.tile(p, NUM_BRAGGS)
    pvec = numpy.concatenate(
        (_aux, numpy.tile(p[0], NUM_DEFECT), numpy.flip(_aux))
    )
    _aux = numpy.tile(d, NUM_BRAGGS)
    dvec = numpy.concatenate(
        ([_aux[0]], [d_2], _aux[2:], numpy.tile(d[3], NUM_DEFECT), numpy.flip(_aux))
    )*numpy.concatenate(([1], alfa**vec1))
    R = numpy.zeros(len(beam.wavelength))
    for i, l in enumerate(beam.wavelength):
        N = numpy.concatenate([
            n_incident[i],
            effective_index_binary_func(n_void[i], n_matrix[i], pvec),
            n_substrate[i],
        ])
        R[i] = _reflectance_spectrum(N, dvec, l, beam.ang_inc_rad, beam.polarisation);
    cost = numpy.mean(numpy.abs(R - ref_experimental))
    return cost


