import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy
import os
import src.utils as u


def generate_harmonic_process(freqs_hz, N_samples, fs_, amplitudes_over_time=None, phases=None,
                              smooth_edges_window=False, mean_harmonic=0.5, var_harmonic=10, corr_factor=1.,
                              local_rng: np.random.Generator = None, scaling_factors=np.array([]),
                              fixed_amplitudes=False):
    """
    Generate a sinusoid with the given frequencies and number of samples.
    The phase is a uniform random variable.
    """

    def amp_generator_wss_lpf(mean_, variance_, max_freq_hz):
        x = local_rng.normal(mean_, variance_, N_samples)
        # Filter with the simplest LPF and cutoff at max_freq_hz
        x = scipy.signal.lfilter(*scipy.signal.butter(4, max_freq_hz / fs_, 'low'), x)
        return x

    num_harmonics_ = freqs_hz.shape[0]
    discrete_times = np.arange(N_samples) / fs_

    if phases is None:
        phases = local_rng.uniform(-np.pi, np.pi, num_harmonics_)

    if scaling_factors.size == 0:
        scaling_factors = local_rng.uniform(0.2, 1.8, num_harmonics_)

    if num_harmonics_ != len(phases) or num_harmonics_ != len(scaling_factors):
        raise ValueError(f"{num_harmonics_ = } must be equal to {len(phases) = } and {len(scaling_factors) = }.")

    if amplitudes_over_time is None:
        # if corr_factor == 0:
        #     # Independent temporal envelopes
        #     amplitudes_over_time = [scaling_factors[idx] * amp_generator_wss_lpf(mean_harmonic, var_harmonic, max_freq_hz=5)
        #                             for idx in range(num_harmonics_)]
        # elif corr_factor == 1:
        #     # SAME temporal envelope, just different rescaling
        #     amplitude_over_time = amp_generator_wss_lpf(mean_harmonic, var_harmonic, max_freq_hz=5)
        #     amplitudes_over_time = [scaling_factors[idx] * amplitude_over_time for idx in range(num_harmonics_)]
        #
        # else:
        amp_gen = lambda: amp_generator_wss_lpf(mean_harmonic, var_harmonic, max_freq_hz=5)
        shared_envelope = amp_gen()
        individual_envelopes = [amp_gen() for _ in range(num_harmonics_)]
        amplitudes_over_time = [scaling_factors[idx] *
                                (corr_factor * shared_envelope + (1. - corr_factor) * individual_envelopes[idx])
                                for idx in range(num_harmonics_)]

        amplitudes_over_time = np.array(amplitudes_over_time)

    # If fixed amplitudes, set all amplitudes to be fixed and equal to the scaling factors
    # If amplitude is np.sqrt(2) and scaling factor is 1, the variance of the signal is 1.
    if fixed_amplitudes:
        amplitudes_over_time = np.ones_like(amplitudes_over_time) * scaling_factors[:, None]

    if freqs_hz.ndim == 1:  # frequency is fixed over time
        z_individual = amplitudes_over_time * np.cos(
            2 * np.pi * freqs_hz[:, None] * discrete_times[None, :] + phases[:, None])
        z = np.sum(z_individual, axis=0)
    else:  # frequency varies over time
        z_individual = np.cos((2 * np.pi * np.cumsum(freqs_hz, axis=1)) / fs_ + phases[:, None]) * amplitudes_over_time
        z = np.sum(amplitudes_over_time * z_individual, axis=0)

    if smooth_edges_window:
        win = scipy.signal.windows.tukey(N_samples, alpha=0.1)
        z = z * win

    return z, amplitudes_over_time


# Design bandpass filter around each harmonic
def bandpass(f_c, fs, bw=20):
    nyq = fs / 2
    sos = scipy.signal.butter(4, [(f_c - bw / 2) / nyq, (f_c + bw / 2) / nyq], btype='band', output='sos')
    return sos


u.set_plot_options(use_tex=True)
parent_dir = "/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/audio/20_05_2025"
file_name = "cello-e2.wav"
# file_name = "Bass.arco.ff.sulC.C1.stereo.aif"
# file_name = "trumpet-G3.wav"
# file_name = "trumpet-C4.wav"
# file_name = "TenorTrombone.ff.A2.stereo.wav"
# file_name = "335559__sonic-wolf__industrial-agitator-recording_short.wav"

x, fs = sf.read(os.path.join(parent_dir, file_name))

if file_name == "cello-e2.wav":
    file_name_plot = 'Cello-E2'
    # f0 = 82.41  # E2
    f0 = 82  # E2
elif file_name == "trumpet-C4.wav":
    file_name_plot = 'Trumpet-C4'
    f0 = 261.63  # C4
elif file_name == "trumpet-G3.wav":
    file_name_plot = 'Trumpet-G3'
    # f0 = 196  # C4
    f0 = 194  # C4
elif file_name == "TenorTrombone.ff.A2.stereo.wav":
    file_name_plot = 'TenorTrombone-A2'
    f0 = 110  # A2
elif file_name == 'Bass.arco.ff.sulC.C1.stereo.aif':
    file_name_plot = 'Bass-C1'
    f0 = 32.7
else:
    file_name_plot = ''
    f0 = 0

if x.ndim > 1:
    x = x[:, 0]

# x = np.random.normal(size=x.shape)
# file_name_plot = 'White noise'

plot_downshifted_harmonics = False
plot_spectrum_ideal_harmonics = True
harmonics = 3
hsize = u.get_plot_width_double_column_latex()
vsize = hsize * 1.5

# Adjust font sizes
suptitle_font_size = 10
title_font_size = 8
label_font_size = 8
tick_font_size = 8

if plot_downshifted_harmonics:

    # fs = 16000
    # x, amps = generate_harmonic_process(np.array([n*f0 for n in range(1, harmonics+1)]), int(fs*4), fs, local_rng=np.random.default_rng(0),
    #                                     mean_harmonic=0, corr_factor=0.9,
    #                                     phases=np.array([0 for n in range(1, harmonics+1)]),
    #                                     scaling_factors=np.array([n/(n+1) for n in range(1, harmonics+1)]))

    # Process each harmonic
    t = np.arange(len(x)) / fs
    outputs = []
    bw = 30  # filter width (Hz)
    harmonic_freqs_hz = []
    for k in range(1, harmonics + 1):
        f_c = k * f0
        harmonic_freqs_hz.append(f_c)

        # 1) band-pass around k·f₀
        sos_bp1 = bandpass(f_c, fs, bw=bw)
        x_bp1 = scipy.signal.sosfilt(sos_bp1, x)

        # 2) shift to f₀ unless we're processing f0 itself
        shift = f_c - f0
        if abs(shift) > 1:
            x_demod = x_bp1 * np.exp(-1j * 2 * np.pi * shift * t)

            # 3) band-pass again around f₀ to isolate
            sos_bp2 = bandpass(f0, fs, bw=bw)
            x_bb = scipy.signal.sosfilt(sos_bp2, x_demod)
        else:
            # first harmonic already at f₀
            x_bb = x_bp1

        outputs.append(np.real(x_bb))

    max_amplitude = np.max(np.abs(np.asarray(outputs)))

    # Plot (2s to 2.3s)
    # start, end = int(2.0593 * fs), int(2.3 * fs)
    start, end = int(6.05 * fs), int(6.55 * fs)
    fig, axes = plt.subplots(harmonics, 1, figsize=(hsize, vsize), sharex=True, constrained_layout=True)

    for i, y in enumerate(outputs):
        time_segment = t[start:end] - t[start]  # Shift time to start at 0
        axes[i].plot(time_segment, y[start:end] / (max_amplitude * 1.001))
        title = f"Harmonic {i + 1}, {harmonic_freqs_hz[i]:.0f}Hz $\\rightarrow$ {harmonic_freqs_hz[0]:.0f}Hz"
        axes[i].set_title(title, fontsize=title_font_size)
        axes[i].grid()
        axes[i].set_ylim([-1, 1])
        axes[i].tick_params(axis='both', labelsize=tick_font_size)

    axes[-1].set_xlabel("Time [s]", fontsize=label_font_size)
    fig.suptitle(f"{file_name_plot.lower().capitalize()}", fontsize=suptitle_font_size)
    fig.show()

    dest_path = os.path.join(os.getcwd(), f"{file_name_plot}_harmonics.pdf")
    fig.savefig(fname=dest_path)
    print(f"Saved to {dest_path = }")

if plot_spectrum_ideal_harmonics:

    for_powerpoint = False

    # Adjust font sizes
    title_font_size = 8
    label_font_size = 8
    tick_font_size = 8
    legend_font_size = 7

    hsize_spec = u.get_plot_width_double_column_latex()
    vsize_spec = hsize_spec*0.4  # paper
    if for_powerpoint:
        vsize_spec = hsize_spec*0.65  # powerpoint

    fft_size = 16384
    x_segment = x[fs:]  # ensure x is long enough
    x_segment = x_segment - np.mean(x_segment)
    hann = scipy.signal.windows.blackmanharris(fft_size,sym=False)
    spectrum = np.log1p(np.abs(scipy.fft.rfft(x_segment[:fft_size]*hann, n=fft_size)))
    freqs = np.fft.rfftfreq(fft_size, 1. / fs)

    f_min = 0
    f_max = f0*25.5  # adjust as needed
    freq_mask = (freqs >= f_min) & (freqs <= f_max)

    fig, ax = plt.subplots(figsize=(hsize_spec, vsize_spec), constrained_layout=True)
    ax.plot(freqs[freq_mask], spectrum[freq_mask], label="Spectrum")

    # Add vertical lines at integer multiples of f0
    harmonics = np.arange(f0, f_max, f0)
    for i, f in enumerate(harmonics):
        if f >= f_min:
            ax.axvline(f, color='r', linestyle='-', linewidth=0.5,
                       label="Ideal harmonics" if i == 0 else "")

    ax.set(yticklabels=[])  # remove the tick labels
    ax.tick_params(left=False)  # remove the ticks
    ax.set_xlabel("Frequency (Hz)", fontsize=label_font_size)
    ax.set_ylabel("Log magnitude", fontsize=label_font_size, labelpad=-5)
    if for_powerpoint:
        ax.set_title(f"Spectrum of {file_name_plot.lower()} and ideal harmonics", fontsize=title_font_size)
    ax.legend(fontsize=legend_font_size)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    fig.show()

    fname_save = os.path.join(os.getcwd(), f"{file_name_plot}_spectrum_vs_harmonics.pdf")
    if for_powerpoint:
        fname_save = os.path.join(os.getcwd(), f"{file_name_plot}_spectrum_vs_harmonics_pp.pdf")

    fig.savefig(fname=fname_save)
    print(f"Saved to {fname_save}")

    # Run pdfcrop
    import subprocess
    cropped_fname = fname_save  # fname_save.replace(".pdf", "_cropped.pdf")
    subprocess.run(["pdfcrop", fname_save, cropped_fname], check=True)
    print(f"Cropped PDF saved to {cropped_fname}")
