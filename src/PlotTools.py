# Contiene varias funciones para graficar los resultados.

import numpy
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib import cm

from TMMOOPapproach import find_closest


def plot_spectrum_1d(
    x_range, S,
    size=(6,4),
    linestyle="-", linewidth=1, linecolor="tab:blue",
    ylabel="Reflectance", xlabel="Wavelength [nm]",
    ):
    '''
        Plot the spectrum of Reflectance, Transmittance and Absorbance as input.
            fig, ax = plot_spectrum_1d(
                x_range, S,
                size=(6,4),
                linestyle="-", linewidth=1, linecolor="tab:blue",
                ylabel="Reflectance", xlabel="Wavelength [nm]",
            )
                x_range: wavelength range or angle of incidence range
                S: spectrum or spectra
                size: size of the figure
    '''
    fig, ax = pyplot.subplots(figsize=size)
    ax.plot(
        x_range, S,
        linestyle=linestyle,
        linewidth=linewidth,
        c=linecolor,
    )
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    return fig, ax


def plot_spectrum_2d(
    x_range, y_range, S,
    size=(6,4), cmap="viridis", shading="auto",
    ylabel="Angle of incidence [degree]", xlabel="Wavelength [nm]",
    title_label="Reflectance",
    ):
    '''
        Plot the contourf of the 2d spectrum of Reflectance, Transmittance and Absorbance as input for wavelength and angle of incidence dependence.
        It can also be used to plot the electromagnetic field in terms of the wavelength and position.
            fig, ax, im = plot_spectrum_2d(
                x_range, y_range, S,
                size=(6,4), cmap="viridis",
                ylabel="Angle of incidence [degree]", xlabel="Wavelength [nm]",
                title_label="Reflectance",
            )
                x_range, y_range: wavelength range or angle of incidence range
                S: spectrum or spectra
                size: size of the figure
    '''
    fig, ax = pyplot.subplots(figsize=size)
    im = ax.pcolormesh(
        x_range, y_range, S,
        cmap=cmap, shading=shading,
    )
    cb = pyplot.colorbar(im)
    cb.set_label(title_label)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    return fig, ax, im


def plot_fit_spectrum(
    x, Xexp, Xmodel,
    size=(6,4),
    linestyle=["-", "--"], linewidth=[1, 1], linecolor=["tab:blue", "tab:orange"],
    labels=["Experimental", "Model"],
    xlabel="Wavelength [nm]",
    ylabel="Reflectance",
    ):
    '''
        Recipe for plotting a comparison of the model and experimental spectra.
            fig, ax = plot_fit_spectrum(
                x, Xexp, Xmodel,
                size=(6,4),
                linestyle=["-", "--"],
                linewidth=[1, 1],
                linecolor=["tab:blue", "tab:orange"],
                labels=["Experimental", "Model"],
                xlabel="Wavelength [nm]",
                ylabel="Reflectance",
            )
                x: range of variable (wavelength or angle)
                Xexp: experimental spectrum
                Xmodel: model spectrum
                labels: labels of the legend
                size: size of the figure
    '''
    fig, ax = pyplot.subplots(figsize=size)
    ax.plot(
        x, Xexp,
        linestyle=linestyle[0],
        linewidth=linewidth[0],
        c=linecolor[0],
        label=labels[0],
    )
    ax.plot(
        x, Xmodel,
        linestyle=linestyle[1],
        linewidth=linewidth[1],
        c=linecolor[1],
        label=labels[1],
    )
    pyplot.xlabel(xlabel)
    pyplot.xlabel(ylabel)
    pyplot.legend(loc="best")
    return fig, ax


def plot_solution_space_2d(
    x_range, y_range, S,
    size=(6,4), cmap="viridis",
    ylabel="Porosity", xlabel="Optical thickness [nm]",
    title_label="Min. error",
    ):
    '''
        Plot the contourf of the 2d error space for the brute-force search as input.
            fig, ax, im = plot_spectrum_2d(
                x_range, y_range, S,
                size=(6,4), cmap="viridis",
                ylabel="Porosity", xlabel="Optical thickness [nm]",
                title_label="Min. error")
    '''
    fig, ax = pyplot.subplots(figsize=size)
    im = ax.pcolormesh(
        x_range, y_range, S,
        cmap=cmap,
    )
    cb = pyplot.colorbar(im)
    cb.set_label(title_label)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    return fig, ax, im


def plot_index_profile(
    results,
    plotemf=False, angle=None, wavelength=None,
    size=(6,4), xlabel="Thickness profile [nm]",
    ylabel="Refractive index at lambda_0",
    cmap_colors="viridis",
    ):
    '''
        Plots the index of refraction ath certain wavelength (usually λ0) of the multilayer structure.

            plot_index_profile(
                results, angle=None, wavelength=None,
                plotemf=False, size=(6,4), xlabel="Thickness profile [nm]",
                ylabel="Refractive index at lambda_0")

                results: structure of the results obtained from the simulation.
                plotemf: whether to overlap the emf for the reference wavelength or not
                angle: single angle of incidence to plot te EMF (if is None, defaults to the mean of beam.angle_inc_deg)
                wavelength: single wavelength to plot the EMF (if is None, defaults to te mean of beam.wavelength)
            
            The EMF is calculated taking into account the average using the beam.polarisation.
    '''

    def _get_colors(num, cmap_colors="viridis"):
        if num < 10:
            return ["tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        else:
            return cm.get_cmap(cmap_colors, num)

    d = results.physical_thickness[1:-1] # remove incident and emergent media
    doffset = 0.05*numpy.sum(d)
    new_d = numpy.hstack([-doffset, numpy.cumsum(numpy.hstack([0.0, d, doffset]))])
    x = new_d[:-1]
    w = numpy.diff(new_d)
    h = results.n_wavelength_0
    block1 = [((x[i], 0.0), w[i], h[i]) for i in range(len(x))]

    fig, ax = pyplot.subplots(figsize=size)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)

    if plotemf:
        if angle is None:
            angle = numpy.mean(results.angle_inc_deg)
        if wavelength is None:
            wavelength = numpy.mean(results.wavelength)
        a = find_closest(angle, results.angle_inc_deg)
        w = find_closest(wavelength, results.wavelength)
        emf = (1 - results.polarisation)*results.EMFp[w, a, :] + results.polarisation*results.EMFs[w, a, :]
        ax.plot(
            results.layers_depth, emf,
            linestyle="-", c="black",
            label=f"EMF at {wavelength} nm")
        pyplot.legend(loc="best")
    else:
        ax.plot([0, numpy.max(new_d)],[0, numpy.max(h)], linestyle=" ")
    
    num_unique = numpy.unique(h)
    len_num_unique = len(num_unique)
    colors = _get_colors(len_num_unique, cmap_colors=cmap_colors)
    for i in range(len_num_unique):
        equal_layers = numpy.where(h == h[i])[0]
        for l in equal_layers:
            ax.add_patch(Rectangle(block1[l][0], block1[l][1], block1[l][2], facecolor=colors[i]))
    return fig, ax


def pbg_dispersion_1d(
    results,
    wave="p", kpart="real",
    size=(6,4), xlim=(0, 1), ylim=(0, 1)):
    '''
        Plots the photonic dispersion (Bloch wavevector) of a photonic crystal structure,
        computed for a range of frequencies (wavelengths) and one angle of incidence.
        Takes only one polarisation. If you want you can pass optionally the part of the
        wavevector to plot. By default, plots the real one (as the imaginary does not represent much).
            pbg_dispersion_1d(results, wave="p", kpart="real", size=(6,4))
            
                results: results structure from the simulation
                wave: either p-wave ("p", default) or s-wave ("s")
                kpart: either real ("real", default) or imginary ("imag")
                size: size of the figure
    '''
    omega = results.omega*results.crystal_period/2.0/numpy.pi # frequency range normalized
    k = results.crystal_period/numpy.pi
    if wave == "p":
        k *= results.bloch_vector_p
    elif wave == "s":
        k *= results.bloch_vector_s
    else:
        raise ValueError("The wave parameter should be either 'p' or 's'.")
    
    if kpart == "real":
        k = numpy.real(k)
    elif kpart == "imag":
        k = numpy.imag(k)
    else:
        raise ValueError("The kpart parameter should be either 'real' or 'imag'.")
    
    fig, ax = pyplot.subplots(figsize=size)
    pyplot.xlabel("K*Lambda/pi")
    pyplot.ylabel("omega*Lambda/(2*pi)")
    ax.plot(k, omega)
    pyplot.xlim(xlim)
    pyplot.ylim(ylim)

    return fig, ax


def pbg_dispersion_1d_alt(
    results,
    kpart="real",
    size=(6,4), xlim=(-1, 1), ylim=(0, 1)
    ):
    """
        Plots the photonic dispersion (Bloch wavevector) of a photonic crystal structure,
        computed for a range of frequencies (wavelengths) and one angle of incidence, for both polarisations. If you want you can pass optionally the part of the
        wavevector to plot. By default, plots the real one (as the imaginary does not represent much).

        pbg_dispersion_1d(results, kpart="real", size=(6,4))
            
                results: results structure from the simulation            
                kpart: either real ("real", default) or imginary ("imag")
                size: size of the figure
    """
    omega = results.omega*results.crystal_period/2.0/numpy.pi # frequency range normalized
    kp = results.bloch_vector_p*results.crystal_period/numpy.pi
    ks = results.bloch_vector_s*results.crystal_period/numpy.pi

    if kpart == "real":
        kp = numpy.real(kp)
        ks = numpy.real(ks)
    elif kpart == "imag":
        ks = numpy.imag(ks)
        kp = numpy.imag(kp)
    else:
        raise ValueError("The kpart parameter should be either 'real' or 'imag'.")
    
    fig, ax = pyplot.subplots(figsize=size)
    pyplot.xlabel("K*Lambda/pi")
    pyplot.ylabel("omega*Lambda/(2*pi)")
    ax.plot(-kp, omega, linestyle="-", label="p-wave")
    ax.plot(ks, omega, linestyle="--", label="s-wave")
    ax.plot([0, 0], [0, 1], linestyle="-", c="black", linewidth=0.5)
    _x = numpy.linspace(-1, 1, 11).round(2)
    ax.set_xticks(_x)
    ax.set_xticklabels(numpy.hstack([-_x[:5], _x[5:]]).astype(str))
    pyplot.xlim(xlim)
    pyplot.ylim(ylim)
    pyplot.legend(loc="best")

    return fig, ax


def pbg_dispersion_1d_imre(
    results,
    wave="p",
    size=(6,4), xlim=(-1, 1), ylim=(0, 1)
    ):
    """
        Plots the photonic dispersion (Bloch wavevector) of a photonic crystal structure,
        computed for a range of frequencies (wavelengths) and one angle of incidence.
        
        Takes one polarisation type in complex format, and plots on the left the imaginary
        part and on the right the real part.

        pbg_dispersion_1d_imre(results, wave="p", size=(6,4))
            
                results: results structure from the simulation            
                wave: either p-wave ("p", default) or s-wave ("s")
                size: size of the figure
    """
    omega = results.omega*results.crystal_period/2.0/numpy.pi # frequency range normalized
    k = results.crystal_period/numpy.pi
    if wave == "p":
        k *= results.bloch_vector_p
    elif wave == "s":
        k *= results.bloch_vector_s
    else:
        raise ValueError("The wave parameter should be either 'p' or 's'.")
    
    fig, ax = pyplot.subplots(figsize=size)
    pyplot.xlabel("K*Lambda/pi")
    pyplot.ylabel("omega*Lambda/(2*pi)")
    ax.plot(-numpy.imag(k), omega, linestyle="-", label="Imag")
    ax.plot(numpy.real(k), omega, linestyle="--", label="Real")
    ax.plot([0, 0], [0, 1], linestyle="-", c="black", linewidth=0.5)
    _x = numpy.linspace(-1, 1, 11).round(2)
    ax.set_xticks(_x)
    ax.set_xticklabels(numpy.hstack([-_x[:5], _x[5:]]).astype(str))
    pyplot.xlim(xlim)
    pyplot.ylim(ylim)
    pyplot.legend(loc="best")

    return fig, ax


def pbg_dispersion_2d(
    results,
    wave="p",
    size=(6,4), s_cmap="Pastel1", p_cmap="Set3", color_lines=None):
    '''
        Plots the photonic dispersion (Bloch wavevector) of a photonic crystal structure,
        computed for a range of frequencies (wavelengths) and a range of angle of incidences.
        This function plots the Bloch wavevector for only one of the polarisation types.
            pbg_dispersion_2d(results, wave="p", size=(6,4))
            
                results: results structure from simulations
                wave: either "p" (p-wave, default) or "s" (s-wave)
                size: size of the figure
    '''

    qz, omega, omega_h, omega_l, kappa = _get_bloch_parameters_2d(
        results, wave=wave,
    )

    if color_lines is None:
        color_lines = ["dimgrey", "darkgrey", "lightgrey"]

    cmap = p_cmap if wave == "p" else s_cmap

    fig, ax = pyplot.subplots(figsize=size)
    im = ax.pcolormesh(
        qz, omega, numpy.real(kappa), cmap=cmap,
    )
    pyplot.xlabel("Parallel wavevector, qz (2*pi/Lambda)")
    pyplot.ylabel("omega*Lambda/(2*pi)")
    pyplot.xlim(0.0, qz[-1])
    pyplot.ylim(0.0, omega[0])

    ax.plot(
        [0.0, omega[0]], [0.0, omega[0]],
        linewidth=1.5, linestyle="-", c=color_lines[0], label="Light line")
    
    ax.plot(
        [0.0, qz[-1]], [omega_h, omega_h],
        linewidth=1.5, linestyle="--", c=color_lines[1], label="omega_h")
    
    ax.plot(
        [0.0, qz[-1]], [omega_l, omega_l],
        linewidth=1.5, linestyle="-.", c=color_lines[2], label="omega_l")
    
    ax.plot([0, 0], [0, 1], linestyle="-", c="black", linewidth=0.5)
    pyplot.legend(loc="best")

    return fig, ax, im


def pbg_dispersion_2d_alt(
    results,
    size=(6,4), s_cmap="Pastel1", p_cmap="Set3", color_lines=None,
    space_title=20):
    '''
        Plots the photonic dispersion (Bloch wavevector) of a photonic crystal structure,
        computed for a range of frequencies (wavelengths) and a range of angle of incidences.
        This function plots the Bloch wavevector for both polarisations.
            pbg_dispersion_2d_alt(results, size=(6,4))
            
                results: results structure from simulations
                size: size of the figure
    '''

    if color_lines is None:
        color_lines = ["dimgrey", "darkgrey", "lightgrey"]
    
    qz, omega, omega_h, omega_l, kappa_s = _get_bloch_parameters_2d(results, wave="s")

    _, _, _, _, kappa_p = _get_bloch_parameters_2d(results)

    fig, ax = pyplot.subplots(figsize=size)
    im = ax.pcolormesh(
        qz, omega, numpy.real(kappa_s), cmap=s_cmap,
    )
    im = ax.pcolormesh(
        -qz, omega, numpy.real(kappa_p), cmap=p_cmap,
    )
    pyplot.xlabel("Parallel wavevector, qz (2*pi/Lambda)")
    pyplot.ylabel("omega*Lambda/(2*pi)")
    pyplot.xlim(-qz[-1], qz[-1])
    pyplot.ylim(0, omega[0])

    ax.plot(
        [0.0, omega[0]], [0.0, omega[0]],
        linewidth=1.5, linestyle="-", c=color_lines[0], label=None)
    ax.plot(
        [0.0, -omega[0]], [0.0, omega[0]],
        linewidth=1.5, linestyle="-", c=color_lines[0], label="Light line")
    
    ax.plot(
        [-qz[-1], qz[-1]], [omega_h, omega_h],
        linewidth=1.5, linestyle="--", c=color_lines[1], label="omega_h")
    
    ax.plot(
        [-qz[-1], qz[-1]], [omega_l, omega_l],
        linewidth=1.5, linestyle="-.", c=color_lines[2], label="omega_l")
    
    ax.plot([0, 0], [0, 1], linestyle="-", c="black", linewidth=0.5)
    pyplot.legend(loc="best")

    p, s = "p-wave", "s-wave"
    title_ = p.ljust(len(p)+space_title) #add four spaces at the beginning
    title_ += s
    pyplot.title(title_)

    return fig, ax, im


def _get_bloch_parameters_2d(results, wave="p"):
    qz, omega_h, omega_l = results.wavevector_qz, results.omega_h, results.omega_l
    L = results.crystal_period # periodicity
    omega = results.omega*L/2.0/numpy.pi # frequency range normalized
    if wave == "p":
        kbloch = results.bloch_vector_p*L # ./ π    
    elif wave == "s":
        kbloch = results.bloch_vector_s*L# ./ π
    else:
        raise ValueError("The wave type must be either 'p' or 's'.")
    # k normalized for angle-frequency dependence
    # logical matrices, used to select points which belong to the forbidden bands
    kappa = numpy.cos(kbloch)#.*π) # rhs equation 45 in the paper, π comes from previous normalization
    kappa[numpy.abs(kappa) < 1.0] = 1.0 # propagating wave
    kappa[numpy.abs(kappa) > 1.0] = 0.0 # evanescent wave

    return qz, omega, omega_h, omega_l, kappa

