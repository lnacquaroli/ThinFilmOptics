import numpy
import functools


class LayerTMMO():
    '''
        Construct the layer information for the simulation.

            LayerTMMO(n, thickness=numpy.nan, layer_type="GT", n_wavelength_0=-eps)

        n: Index of refraction 
        thickness: Thickness in nm for the "GT". Defaults to nan for semiinfinite layers
        layer_type: "GT" (geometrical thickness), or "OT" (optical thickness)
        n_wavelength_0: Central or reference lambda (mostly for multilayer structures)
            If it is not declared, it will default to numpy.mean(beam.wavelength)
    '''
    def __init__(
        self, n,
        thickness=numpy.nan, layer_type="GT", n_wavelength_0=-numpy.finfo(float).eps
    ):
        self.index_refraction = n
        self.thickness = thickness
        self.layer_type = layer_type
        self.n_wavelength_0 = n_wavelength_0


class TMMOptics():
    '''
        Defines the structures for the results.
    '''

    def __init__(self) -> None:
        pass
    
    def beam_parameters(
        self, wavelength,
        angle_incidence=0.0, polarisation=0.5, wavelength_0=-numpy.finfo(float).eps,
        ) -> None:  
        assert(0.0 <= polarisation <= 1.0), "the polarisation should be between 0 and 1"
        if isinstance(wavelength, int) or isinstance(wavelength, float) or isinstance(wavelength, range) or isinstance(wavelength, list):
            wavelength = numpy.array([wavelength]).reshape(-1)
        assert(wavelength.all() > 0.0), "the wavelength should be > 0"
        if isinstance(angle_incidence, int) or isinstance(angle_incidence, float) or isinstance(angle_incidence, range) or isinstance(angle_incidence,list):
            angle_incidence = numpy.array([angle_incidence]).reshape(-1)
        assert(0.0 <= angle_incidence.all() <= 90.0), "the angle of incidence should be between 0 and 90 degrees"  
        self.wavelength = wavelength
        self.angle_inc_deg = angle_incidence
        self.angle_inc_rad = angle_incidence*numpy.pi/180.0
        self.polarisation = polarisation

        # Check if wavelength_0 is inside beam.wavelength
        self.wavelength_0 = _check_wavelength_0(self, wavelength_0)

        # Find beam.wavelength closest to wavelength_0
        self.wavelength_0_idx = find_closest(wavelength_0, self.wavelength)
    
    def tmm_spectra(self, layers):
        '''
            Calculates the reflectance and transmittance spectra.
        '''
        
        # Build the sequence of n and d depending on the input
        self = _add_gt_thickness(self, layers)

        self = _transfer_matrix_spectra(self, layers)

        return self

    def tmm_emf(self, layers, layers_split=10):

        '''
            Calculates the electromagnetic field distribution.
        '''
        
        # Build the sequence of n and d depending on the input
        self = _add_gt_thickness(self, layers)

        self = _transfer_matrix_emf(self, layers, layers_split)
        self = _layers_depth(self, layers, layers_split)

        return self

    def tmm_spectra_emf(self, layers, layers_split=10):

        '''
            Calculates the reflectance and transmittance spectra, and also the electromagnetic field distribution.
        '''
        
        # Build the sequence of n and d depending on the input
        self = _add_gt_thickness(self, layers)

        self = _transfer_matrix_spectra_emf(self, layers, layers_split)
        self = _layers_depth(self, layers, layers_split)

        return self

    def photonic_dispersion(self, layers):

        assert(len(layers) > 3), "the number of layers must be greater than 3."
        
        d = self.physical_thickness[1:3]
        n = [
            layers[1].index_refraction[self.wavelength_0_idx],
            layers[2].index_refraction[self.wavelength_0_idx],
        ]
        self.crystal_period = numpy.sum(d)
        self.wavevector_qz = numpy.sin(self.angle_inc_rad)*numpy.pi/2.0 # parallel wavevector qz
        self = _photonic_dispersion(self, d, n)
        
        n0, n1, n2 = self.n_wavelength_0[0:3]
        self.omega_h = self.crystal_period/numpy.pi/(d[0]*n1 + d[1]*n2)*numpy.arccos(-numpy.abs(n1 - n2)/(n1 + n2))
        self.omega_l = self.crystal_period/numpy.pi/(d[1]*numpy.sqrt(n2**2 - n0**2) + d[0]*numpy.sqrt(n1**2 - n0**2))*numpy.arccos(numpy.abs((n1**2*numpy.sqrt(n2**2 - n0**2) - n2**2*numpy.sqrt(n1**2 - n0**2))/(n1**2*numpy.sqrt(n2**2 - n0**2) + n2**2*numpy.sqrt(n1**2 - n0**2))))

        return self


def _check_wavelength_0(self, wavelength_0):
    if wavelength_0 == -numpy.finfo(float).eps:
        return numpy.mean(self.wavelength)
    else:
        return wavelength_0


def _add_gt_thickness(self, layers):
    '''
        Build physical thicknesses and n_wavelength_0 depending on the input. It follows some logic depending on whether n_wavelength_0 was input.
    '''
    n_wavelength_0 = numpy.zeros(len(layers))
    physical_thickness = numpy.zeros(len(layers))
    for i, x in enumerate(layers):
        # n_wavelength_0 depending on the input
        if x.n_wavelength_0 == -numpy.finfo(float).eps: # not specified
            n_wavelength_0[i] = numpy.real(x.index_refraction[self.wavelength_0_idx])
        else:
            n_wavelength_0[i] = numpy.real(x.n_wavelength_0)
        # Build thickness depending on the input
        physical_thickness[i] = x.thickness
        if x.layer_type == "OT":
            physical_thickness[i] *= self.wavelength_0/n_wavelength_0[i]
    self.n_wavelength_0 = n_wavelength_0
    self.physical_thickness = physical_thickness
    return self


def _layers_depth(self, layers, h):
    # Provide the multilayer depth considering the h division
    d = self.physical_thickness[1:-1]
    d = d.reshape(len(d), 1)
    l = (d/h)*numpy.ones((1, h)) # outer product
    l = numpy.insert(l, 0, 0)
    l = numpy.cumsum(l)[:-1] # remove last from cumsum
    self.layers_depth = l
    return self


def _transfer_matrix_spectra(self, layers):
    '''
        Computes the reflection and transmission coefficients and spectra with the transfer matrix method.
    '''
    # Initialize variables
    len_ang_inc, len_wavelength = len(self.angle_inc_rad), len(self.wavelength)
    numlay = len(layers) - 1 # number of layers in the structure
    ts = numpy.zeros((len_wavelength, len_ang_inc), dtype=complex)
    tp, rs, rp = ts.copy(), ts.copy(), ts.copy()
    delta = numpy.zeros((len_wavelength, len_ang_inc, numlay + 1), dtype=complex)
    adm_p, adm_s = delta.copy(), delta.copy()
    Ms, Mp = numpy.eye(2, dtype=complex), numpy.eye(2, dtype=complex)
    I = numpy.eye(2, dtype=complex)[:, :, numpy.newaxis]
    cosphi = numpy.zeros(numlay + 1, dtype=complex)
    n = cosphi.copy()
    for l, wavelen in enumerate(self.wavelength):
        for a in range(len_ang_inc):
            cosphi[0] = numpy.cos(self.angle_inc_rad[a])
            adm_s[l, a, :], adm_p[l, a, :], delta[l, a, :], Ms, Mp = _complete_transfer_matrix(self.physical_thickness, cosphi, layers, l, wavelen, I, n)
            # calculation of the spectra
            rs[l, a], rp[l, a], ts[l, a], tp[l, a] = _r_t_coefficients(
                adm_s[l, a, 0], adm_s[l, a, numlay], Ms,
                adm_p[l, a, 0], adm_p[l, a, numlay], Mp,
            )
    self.ref_coeff_p, self.ref_coeff_s, self.tra_coeff_p, self.tra_coeff_s = rp, rs, tp, ts
    self.transmittance_p = numpy.real(adm_p[:, :, 0]*adm_p[:, :, -1])*numpy.abs(tp)**2
    self.transmittance_s = numpy.real(adm_s[:, :, 0]*adm_s[:, :, -1])*numpy.abs(ts)**2
    self.reflectance_p, self.reflectance_s = numpy.abs(rp)**2, numpy.abs(rs)**2
    self.reflectance = (1.0 - self.polarisation)*self.reflectance_s + self.polarisation*self.reflectance_p
    self.transmittance = (1.0 - self.polarisation)*self.transmittance_s + self.polarisation*self.transmittance_p    
    self.admittance_p, self.admittance_s, self.phase = adm_p, adm_s, delta
    
    return self


def _transfer_matrix_emf(self, layers, h):
    '''
        Computes the EMF with the transfer matrix method.
    '''
    # Initialize variables
    len_ang_inc, len_wavelength = len(self.angle_inc_rad), len(self.wavelength)
    numlay = len(layers) - 1 # number of layers in the structure
    delta = numpy.zeros((len_wavelength, len_ang_inc, numlay + 1), dtype=complex)
    adm_p, adm_s = delta.copy(), delta.copy()
    Ms, Mp = numpy.eye(2, dtype=complex), numpy.eye(2, dtype=complex)
    I = numpy.eye(2, dtype=complex)[:, :, numpy.newaxis]
    cosphi = numpy.zeros(numlay + 1, dtype=complex)
    n = cosphi.copy()
    emfs = numpy.zeros((len_wavelength, len_ang_inc, (len(layers) - 2)*h))
    emfp = emfs.copy()
    for l, wavelen in enumerate(self.wavelength):
        for a in range(len_ang_inc):
            cosphi[0] = numpy.cos(self.angle_inc_rad[a])
            adm_s[l, a, :], adm_p[l, a, :], delta[l, a, :], Ms, Mp = _complete_transfer_matrix(self.physical_thickness, cosphi, layers, l, wavelen, I, n)
            emfs[l, a, :] = _emfield(delta[l, a, :], adm_s[l, a, :], Ms, numlay + 1, h)
            emfp[l, a, :] = _emfield(delta[l, a, :], adm_p[l, a, :], Mp, numlay + 1, h)
    self.admittance_p, self.admittance_s, self.phase = adm_p, adm_s, delta
    self.EMFp, self.EMFs = emfp, emfs
    
    return self


def _transfer_matrix_spectra_emf(self, layers, h):
    '''
        Computes the reflection and transmission coefficients and spectra, and the EMF with the transfer matrix method.
    '''
    # Initialize variables
    len_ang_inc, len_wavelength = len(self.angle_inc_rad), len(self.wavelength)
    numlay = len(layers) - 1 # number of layers in the structure
    ts = numpy.zeros((len_wavelength, len_ang_inc), dtype=complex)
    tp, rs, rp, r, t = ts.copy(), ts.copy(), ts.copy(), ts.copy(), ts.copy()
    delta = numpy.zeros((len_wavelength, len_ang_inc, numlay + 1), dtype=complex)
    adm_p, adm_s = delta.copy(), delta.copy()
    Ms, Mp = numpy.eye(2, dtype=complex), numpy.eye(2, dtype=complex)
    I = numpy.eye(2, dtype=complex)[:, :, numpy.newaxis]
    cosphi = numpy.zeros(numlay + 1, dtype=complex)
    n = cosphi.copy()
    emfs = numpy.zeros((len_wavelength, len_ang_inc, (len(layers) - 2)*h))
    emfp = emfs.copy()
    for l, wavelen in enumerate(self.wavelength):
        for a in range(len_ang_inc):
            cosphi[0] = numpy.cos(self.angle_inc_rad[a])
            adm_s[l, a, :], adm_p[l, a, :], delta[l, a, :], Ms, Mp = _complete_transfer_matrix(self.physical_thickness, cosphi, layers, l, wavelen, I, n)
            # calculation of the spectra
            rs[l, a], rp[l, a], ts[l, a], tp[l, a] = _r_t_coefficients(
                adm_s[l, a, 0], adm_s[l, a, numlay], Ms,
                adm_p[l, a, 0], adm_p[l, a, numlay], Mp,
            )
            emfs[l, a, :] = _emfield(delta[l, a, :], adm_s[l, a, :], Ms, len(layers), h)
            emfp[l, a, :] = _emfield(delta[l, a, :], adm_p[l, a, :], Mp, len(layers), h)
    self.ref_coeff_p, self.ref_coeff_s, self.tra_coeff_p, self.tra_coeff_s = rp, rs, tp, ts
    self.transmittance_p = numpy.real(adm_p[:, :, 0]*adm_p[:, :, -1])*numpy.abs(tp)**2
    self.transmittance_s = numpy.real(adm_s[:, :, 0]*adm_s[:, :, -1])*numpy.abs(ts)**2
    self.reflectance_p, self.reflectance_s = numpy.abs(rp)**2, numpy.abs(rs)**2
    self.reflectance = (1.0 - self.polarisation)*self.reflectance_s + self.polarisation*self.reflectance_p
    self.transmittance = (1.0 - self.polarisation)*self.transmittance_s + self.polarisation*self.transmittance_p    
    self.admittance_p, self.admittance_s, self.phase = adm_p, adm_s, delta
    self.EMFp, self.EMFs = emfp, emfs
    
    return self


def _complete_transfer_matrix(d, cosphi, layers, l, wavelen, I, n):
    numlay = len(layers)
    n[0] = layers[0].index_refraction[l]
    for c in range(1, numlay):
        n[c] = layers[c].index_refraction[l]
        # compute angles inside each layer according to the Snell law
        cosphi[c] = _snell_cosine(n[c - 1], n[c], cosphi[c - 1])
    # phase shifts for each layer: 2*pi = 6.283185307179586
    delta = (6.283185307179586*n*d*cosphi/wavelen).reshape(-1)
    adm_s, adm_p = n*cosphi, n/cosphi
    # total transfer matrix
    Ms = functools.reduce(
        numpy.matmul, numpy.append(I, _tmatrix(delta[1:-1], adm_s[1:-1]), 2).T).T
    Mp = functools.reduce(
        numpy.matmul, numpy.append(I, _tmatrix(delta[1:-1], adm_p[1:-1]), 2).T).T
    Ms[0, 0], Ms[1, 1] = Ms[1, 1], Ms[0, 0]
    Mp[0, 0], Mp[1, 1] = Mp[1, 1], Mp[0, 0]
    return adm_s, adm_p, delta, Ms, Mp


def _snell_cosine(n1, n2, cosphi):
    '''
	    Snell's law in cosine form. Returns the cosine already.
    '''
    x = numpy.sqrt(1.0 - (n1/n2)**2*(1.0 - cosphi**2))
    return x


def _tmatrix(beta, adm):
    # Optic transfer matrix (iternal function)
    cos_beta, sin_beta = numpy.cos(beta), -1j*numpy.sin(beta)
    Q = numpy.array([
        [cos_beta, sin_beta/adm],
        [adm*sin_beta, cos_beta],
    ])
    return Q


def _inverse_tmatrix(beta, adm):
    # Inverse of optic transfer matrix (iternal function)
    cos_beta, sin_beta = numpy.cos(beta), 1j*numpy.sin(beta)
    Xi = numpy.array([
        [cos_beta, sin_beta/adm],
        [adm*sin_beta, cos_beta],
    ])
    return Xi


def _r_t_coefficients(adm_s_0, adm_s_m, Ms, adm_p_0, adm_p_m, Mp):
    '''
        Computes the reflection and transmission coefficients given the admittance and transfer matrix of the whole structure per wavelenth and angle of incidence.
    '''
    b, c, d = Ms[1,0]/adm_s_0, Ms[0,1]*adm_s_m, Ms[1,1]*adm_s_m/adm_s_0
    rs = (Ms[0,0] - b + c - d)/(Ms[0,0] + b + c + d)
    b, c, d = Mp[1,0]/adm_p_0, Mp[0,1]*adm_p_m, Mp[1,1]*adm_p_m/adm_p_0
    rp = (Mp[0,0] - b + c - d)/(Mp[0,0] + b + c + d)
    ts = 2.0/(adm_s_0*Ms[0,0] + Ms[1, 0] + adm_s_0*adm_s_m*Ms[0, 1] + adm_s_m*Ms[1, 1])
    tp = 2.0/(adm_p_0*Mp[0,0] + Mp[1, 0] + adm_p_0*adm_p_m*Mp[0, 1] + adm_p_m*Mp[1, 1])
    return rs, rp, ts, tp


def _emfield(delta, adm, M, numlay, h):
    m0 = numpy.zeros((2, 2), dtype=complex)
    m1 = numpy.eye(2, 2, dtype=complex) # Identity 2x2 matrix
    g11 = numpy.zeros((numlay - 2)*h, dtype=complex)
    g12 = g11.copy()
    # Divide the phase shift by h but keep η as is for each layer
    m_delta = delta/h
    for c in range(1, numlay - 1):
        _m1 = _inverse_tmatrix(m_delta[c], adm[c])
        for j in range(h):
            k = h*(c - 1) + j
            m1 = numpy.matmul(_m1, m1)
            m0 = numpy.matmul(m1, M)
            g11[k] = m0[0, 0]
            g12[k] = m0[0, 1]
    fi = _field_intensity(g11, g12, adm[0], adm[-1], M)
    return fi


def _field_intensity(g11, g12, adm_0, adm_m, M):
    '''
        Compute the field intensity.
    '''
    fi = numpy.abs((g11 + adm_m*g12)/(0.25*(adm_0*M[0, 0] + M[1, 0] + adm_0*adm_m*M[0,1] + adm_m*M[1,1])))**2
    return fi


'''def _photonic_dispersion(self, d, n):

    def _adm_factor(adm_1, adm_2):
        x = 0.5*(adm_1**2 + adm_2**2)/adm_1/adm_2
        return x
    
    def _bloch_wavevector(a1, a2, f):
        x = numpy.cos(a1)*numpy.cos(a2) - f*numpy.sin(a1)*numpy.sin(a2)
        return x

    def _remove_nans(kappa):
        knan = numpy.isnan(kappa)
        kappa[knan] = kappa[numpy.logical_not(knan)].max()
        return kappa

    self.bloch_vector_p = numpy.ones((len(self.wavelength), len(self.angle_inc_rad)), dtype=complex)
    self.bloch_vector_s = numpy.ones((len(self.wavelength), len(self.angle_inc_rad)), dtype=complex)
    self.omega = 2.0*numpy.pi/self.wavelength # Angular frequency
    # Angle of incidence of the second layer with Snell's law of cosine
    cosphi_1 = numpy.cos(self.angle_inc_rad)
    cosphi_2 = numpy.array([_snell_cosine(n[0], n[1], a) for a in cosphi_1])
    # Prefactor for Bloch wavevector
    factor_s = _adm_factor(n[0]*cosphi_1, n[1]*cosphi_2)
    factor_p = _adm_factor(n[0]/cosphi_1, n[1]/cosphi_2)
    # Bloch wavevectors
    for a in range(len(cosphi_1)):
        for b in range(len(self.omega)):
            a1 = d[0]*self.omega[b]*n[0]*cosphi_1[a]
            a2 = d[1]*self.omega[b]*n[1]*cosphi_2[a]
            self.bloch_vector_p[b, a] = numpy.arccos(_bloch_wavevector(a1, a2, factor_p[a]))
            self.bloch_vector_s[b, a] = numpy.arccos(_bloch_wavevector(a1, a2, factor_s[a]))
    self.bloch_vector_p /= self.crystal_period
    self.bloch_vector_s /= self.crystal_period
    #kappa_p = _remove_nans(kappa_p)
    #kappa_s = _remove_nans(kappa_s)
    return self'''


# I split into real and imag because the arccos seems to have a problem with complexes. It is better but not solved this way.
def _photonic_dispersion(self, d, n):

    def _adm_factor(adm_1, adm_2):
        x = 0.5*(adm_1**2 + adm_2**2)/adm_1/adm_2
        return x
    
    def _bloch_wavevector(a1, a2, f):
        x = numpy.cos(a1)*numpy.cos(a2) - f*numpy.sin(a1)*numpy.sin(a2)
        return x

    def _remove_nans(kappa):
        knan = numpy.isnan(kappa)
        kappa[knan] = kappa[numpy.logical_not(knan)].max()
        return kappa

    kpr = numpy.ones((len(self.wavelength), len(self.angle_inc_rad)))
    kpi, ksr, ksi = kpr.copy(), kpr.copy(), kpr.copy() 
    self.omega = 2.0*numpy.pi/self.wavelength # Angular frequency
    # Angle of incidence of the second layer with Snell's law of cosine
    cosphi_1 = numpy.cos(self.angle_inc_rad)
    cosphi_2 = numpy.array([_snell_cosine(n[0], n[1], a) for a in cosphi_1])
    # Prefactor for Bloch wavevector
    factor_s = _adm_factor(n[0]*cosphi_1, n[1]*cosphi_2)
    factor_p = _adm_factor(n[0]/cosphi_1, n[1]/cosphi_2)
    fsr, fsi = numpy.real(factor_s), numpy.imag(factor_s)
    fpr, fpi = numpy.real(factor_p), numpy.imag(factor_p)
    # Bloch wavevectors
    for a in range(len(cosphi_1)):
        for b in range(len(self.omega)):
            a1 = d[0]*self.omega[b]*n[0]*cosphi_1[a]
            a2 = d[1]*self.omega[b]*n[1]*cosphi_2[a]
            kpr[b, a] = numpy.arccos(_bloch_wavevector(a1, a2, fpr[a]))
            ksr[b, a] = numpy.arccos(_bloch_wavevector(a1, a2, fsr[a]))
            kpi[b, a] = numpy.arccos(_bloch_wavevector(a1, a2, fpi[a]))
            ksi[b, a] = numpy.arccos(_bloch_wavevector(a1, a2, fsi[a]))
    kp = kpr + kpi*1j
    ks = ksr + ksi*1j
    kp, ks = _remove_nans(kp), _remove_nans(ks)
    self.bloch_vector_p = kp / self.crystal_period
    self.bloch_vector_s = ks / self.crystal_period

    return self


def find_closest(a, x):
    '''
        Returns the index of the value in the 1d-array x closest to the scalar value a.

            find_closest(a, x)
    '''
    i = numpy.where(numpy.min(numpy.abs(a - x)) == numpy.abs(a - x))[0][0]
    return i