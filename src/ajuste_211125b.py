
# Ajuste de espesor y porosidad de una capa simple de SP

import numpy
from matplotlib import pyplot
from scipy.optimize import minimize

from EffectiveIndexModels import looyenga
from RefractiveIndexDB import air, silicon
import FitUtils

def recalcula_reflectancia(
    resultados, n_poros, n_solido, n_incidente, n_sustrato,
    ema_func=FitUtils.looyenga,
):
    n_fit_errors = ema_func(n_poros, n_solido, resultados.x_[0])
    r_fit_errors = FitUtils.reflectance(
        numpy.stack((n_incidente, n_fit_errors, n_sustrato), axis=1),
        [resultados.x_[1]],
        beam,
    )
    return r_fit_errors

##### Datos

PATH = "/Users/kwazii/Leandro/code_dev__/python/ajuste_capas_simples"

# Genero una estructura con BeamParameters a partir de los parametros de la luz
LAMBDA = numpy.arange(400, 1000) # rango de longitudes de onda en nm
THETA = 5.0; # grados
POLARIZACION = 0.5; # polarizacion: 1 (p-TM), 0 (s-TE)

beam = FitUtils.BeamParameters(LAMBDA, THETA, POLARIZACION)

# Factor de correccion
FC = 0.8

# Indices de refraccion incidence y sustrato
n_incidente = air(beam.wavelength)
n_sustrato = silicon(beam.wavelength)

# Indices de refraccion de los componentes 1 (poros) y 2 (solido) de looyenga
n_poros = air(beam.wavelength)
n_solido = silicon(beam.wavelength)

# Espectro de reflectancia de la muestra y referencia medidos
r_medido = numpy.loadtxt(PATH + "/251121b00007.txt", delimiter='\t')
r_referencia = numpy.loadtxt(PATH + "/251121b00000.txt", delimiter='\t')
r_referencia[:, 1] *= FC

# Espectro de reflectancia de referencia teorico
r_teorico = FitUtils.reflectance_silicon(beam)

# Cotas minimas y maximas de porosidades y espesores opticos en nanometros
LB, UB = [0.4, 11700], [0.6, 12200]

# Semillas (si no se usa el espacio de soluciones)
#p_0, d_0 = 0.65, 8000

# Para calcular el espacio de soluciones (True) o no (False)
FLAG_1 = True

##### Calculos

# Normaliza el espectro medido
r_medido_norm = FitUtils.normalize_experimental_reflectance(
    r_medido, r_referencia, r_teorico, beam,
)

#pyplot.plot(beam.wavelength, r_medido)
#pyplot.show()

# Calcula el espacio de soluciones (looyenga por default) y lo grafica
error_surface, fun_val, x_0, p_vec, d_vec = FitUtils.solution_space_1(
    FLAG_1,
    LB, UB,
    beam,
    r_medido_norm,
    n_incidente,
    n_sustrato,
    n_poros,
    n_solido,
)

""" fig_errores, ax_errores = pyplot.subplots() 
ax_cbar = ax_errores.contourf(d_vec, p_vec, E_0)
ax_errores.set_xlabel("Espesor optico [nm]")
ax_errores.set_ylabel("Porosidad")
cbar = pyplot.colorbar(ax_cbar)
cbar.set_label("Error")
pyplot.show() """

# Optimiza con algun algoritmo: "nelder-mead", "powell", "BFGS"
metodo = "nelder-mead"

# Argumentos constantes a pasar a la funcion merito
arguments = (
    r_medido_norm,
    n_incidente,
    n_sustrato,
    beam,
    n_poros,
    n_solido,
    FitUtils.looyenga,
    FitUtils.inverse_looyenga,
)

bounds = [(i, j) for i, j in zip(LB, UB)]

resultados = minimize(
    FitUtils.merito1,
    x_0,
    method=metodo,
    options={'xatol': 1e-8, 'disp': True},
    args=arguments,
    bounds=bounds,
)

# Agrego al dictionary the resultados una variable x_ con el espesor fisico y la porosidad
resultados.x_ = numpy.array([
    resultados.x[0],
    FitUtils.inverse_looyenga(resultados.x),
])

# Calculo la reflectancia con los parametros optimos
r_fit = recalcula_reflectancia(
    resultados, n_poros, n_solido, n_incidente, n_sustrato,
)

fig_fit, ax_fit = pyplot.subplots()
ax_fit.plot(beam.wavelength, r_fit, label="Ajuste")
ax_fit.plot(beam.wavelength, r_medido_norm, label="Datos")
ax_fit.set_title(
    f"Porosidad:{resultados.x_[0]: .4f}, Espesor:{resultados.x_[1]: .1f} nm\n" \
    f"Error min.:{resultados.fun: .6f}\n" \
    f"Factor de correcion:{FC: .4f}",
    fontsize=10,
)
ax_fit.legend(loc="best")
ax_fit.set_xlabel("Longitud de onda [nm]")
ax_fit.set_ylabel("Reflectancia")
pyplot.show()
