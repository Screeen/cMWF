"""
This script is to check cyclic beamforming on speech data.
The idea is to build, for each spectral frequency, all different frequency shifted versions and correlate them.
Based on this big correlation matrix, we can find the MVDR beamformer for each spectral frequency.

The script is divided in the following steps:
1. Generation of synthetic data
    1. Load one speech sample from North texas database
    2. Load one impulse response from the database
    3a. Generate a random noise signal
    3b. Rescale the noise signal to the desired SNR
    4a. Generate the desired target signal by convolution of the speech sample and the impulse response
    4b. Convert all signals to the frequency domain
    5. Compute frequency resolution and frequency shifted versions of the IR


2. Estimation of the spectral correlation function
    1. Create all the frequency shifted versions of the target signal
    2. Compute the spatial/cyclic spectral correlation function for each spectral frequency
    3. (Optional) Plot the spatial/cyclic spectral correlation function for each spectral frequency

3. Beamforming
    1. Compute the cyclic MVDR beamformer for each spectral frequency
    1b. Compute the standard MVDR beamformer for each spectral frequency
    2. Apply the beamformers to the frequency domain signals
    3. Convert the beamformed signals to the time domain

4. Evaluation and comparison
    1. Compute the SNRs of all signals: noisy, beamformed (cyclic), beamformed (standard)
    2. (Optional) Plot the noisy and beamformed signals in the time domain
    3. (Optional) Plot the noisy and beamformed signals in the time-frequency domain
    4. Compare the SNRs of the beamformed signals with the noisy signal
    5. (Optional) Save the results to a file
"""
from pathlib import Path

import librosa
import numpy as np
import scipy.signal

from src import manager, utils as u
from src import globs as gs
gs.rng = gs.compute_rng()
from src import f0_estimator
from src import f0_manager
from archive import spectral_correlation_estimator as sce

f0_est = f0_estimator.F0Estimator()
f0_man = f0_manager.F0Manager()

# Add the path to the cyclic beamforming package
module_parent = Path(__file__).resolve().parent

u.set_printoptions_numpy()

fs = 16000
N = -1
M = 3  # Number of microphones

win_name = 'hann'
dft_props = {'nfft': 1024, 'fs': fs, 'nw': 1024}
dft_props['noverlap'] = np.ceil(2 * dft_props['nw'] / 3).astype(int)
dft_props['r_shift_samples'] = dft_props['nw'] - dft_props['noverlap']
dft_props['delta_f'] = dft_props['fs'] / dft_props['nfft']
win = scipy.signal.windows.get_window(win_name, dft_props['nw'], fftbins=True)
window = win / np.sqrt(np.sum(win ** 2))  # normalize to unit power
nfft_real = dft_props['nfft'] // 2 + 1

if win_name == 'cosine' and dft_props['noverlap'] != dft_props['nw'] // 2:
    raise ValueError(f'For {win_name = }, {dft_props["noverlap"] = } must be {dft_props["nw"] // 2 = }')


def local_stft(y_, real_data=False):
    if real_data:
        return_onesided = True
    else:
        return_onesided = False

    _, _, Y_ = scipy.signal.stft(y_, fs=fs, window=window, nperseg=dft_props['nw'], noverlap=dft_props['noverlap'],
                                 detrend=False, return_onesided=return_onesided, boundary=None, padded=False, axis=-1)
    return Y_


def local_istft(Y_, real_data=False):
    if real_data:
        input_onesided = True
    else:
        input_onesided = False
    _, y_ = scipy.signal.istft(Y_, fs=fs, window=window, nperseg=dft_props['nw'], noverlap=dft_props['noverlap'],
                               nfft=dft_props['nfft'], input_onesided=input_onesided)
    return y_


def calculate_harmonic_frequencies(f0_range, num_harmonics, freq_resolution, max_frequency):
    """ Calculate all spectral and cyclic frequencies that correspond to harmonics of f0. """

    f0_range_bins = np.array(f0_range) / freq_resolution
    f0_range_bins = np.r_[
        int(np.floor(f0_range_bins[0])), int(np.ceil(f0_range_bins[-1]))]  # rounded below and above
    # f0_mean_bins = int(np.round(f0_range_bins[1]))

    harmonic_bins_ = []
    for hh in range(num_harmonics + 1):
        harmonic_bins_hh = np.arange(f0_range_bins[0] * hh, f0_range_bins[1] * hh + 1)
        # harmonic_bins_hh = np.arange(f0_mean_bins * hh - 2, f0_mean_bins * hh + 2)
        harmonic_bins_.extend(harmonic_bins_hh)
    harmonic_bins_ = np.unique(np.array(harmonic_bins_))

    # Keep only bins that are in range [0, max_frequency / freq_resolution]
    harmonic_bins_ = harmonic_bins_[harmonic_bins_ >= 0]
    harmonic_bins_ = harmonic_bins_[harmonic_bins_ <= max_frequency / freq_resolution]

    return harmonic_bins_


m = manager.Manager()

# 1. Generation of synthetic data
# Load the speech sample
project_path = Path(__file__).resolve().parent.parent
datasets_path = project_path / 'audio'
pp = datasets_path / 'long_a.wav'
datasets_path = project_path.parent / 'datasets'
# speech_dataset_path = datasets_path / 'north_texas_vowels' / 'data'
# speech_file_name = 'kadpal03.wav'
# if speech_dataset_path.exists():
#     pp = speech_dataset_path / speech_file_name
# else:
#     raise FileNotFoundError(f"Path {speech_dataset_path} does not exist. ")

dry_speech, dry_speech_fs = u.load_audio_file(pp, fs_=fs, N_num_samples=N, offset_seconds=0.045, smoothing_window=False)

# Load the impulse response
ir_dataset_path = datasets_path / 'Hadad_shortcut'
speech_ir, speech_ir_fs = librosa.load(ir_dataset_path / '1.wav', sr=fs, mono=False)
speech_ir = speech_ir[:M, :int(0.75 * fs)]  # Take only the first 0.5 seconds of the impulse response
# speech_ir = speech_ir[:M]
normalization = np.sum(speech_ir ** 2)
speech_ir = speech_ir / normalization  # Normalize the impulse response to unit energy

noise_ir, noise_ir_fs = librosa.load(ir_dataset_path / '2.wav', sr=fs, mono=False)
noise_ir = noise_ir[:M, :int(0.75 * fs)]  # Take only the first 0.5 seconds of the impulse response
# noise_ir = noise_ir[:M]
noise_ir = noise_ir / normalization  # Normalize the impulse response to unit energy

# Check if the sampling rates are the same
if dry_speech_fs != speech_ir_fs:
    raise ValueError(f"Sampling rates are different: {dry_speech_fs = }, {speech_ir_fs = }")

wet_speech = scipy.signal.convolve(dry_speech[np.newaxis, :], speech_ir, mode='full')[:, :len(dry_speech)]
# u.plot([dry_speech, speech_ir, x_temp], ['dry speech', 'impulse response', 'convolved signal'])

# Add noise to the signal
# snr = 30
# noisy_speech, noise = m.add_noise_snr(wet_speech, snr, fs=fs)
noise_dry = gs.rng.normal(0, 1, size=dry_speech.shape) / 20
noise_wet = scipy.signal.convolve(noise_dry[np.newaxis, :], noise_ir, mode='full')[:, :len(dry_speech)]
noisy_speech = wet_speech + noise_wet

# Compute the frequency domain signals
noisy_speech_stft = local_stft(noisy_speech, real_data=True)
dry_speech_stft = local_stft(dry_speech, real_data=True)
wet_speech_stft = local_stft(wet_speech, real_data=True)
speech_atf = np.fft.rfft(speech_ir, n=dft_props['nfft'])

# wet_speech_approx = local_istft(dry_speech_stft[np.newaxis] * atf[..., np.newaxis]).real
sig_prop = {'f_max_hz': 2000, 'alpha_max_hz': 300}
sig_prop['f0_range'], sig_prop['f0_over_time'] = f0_est.find_f0_from_recording(dry_speech, fs, dft_props['r_shift_samples'],
                                                                          dft_props['nfft'])
sig_prop['num_harmonics'] = int(np.ceil(sig_prop['f_max_hz'] / sig_prop['f0_range'][1]))

# Calculate spectral and cyclic frequencies (bins) which fall on harmonics of f0
spectral_bins_harmonics = f0_man.get_harmonics_as_multiples_of_f0(sig_prop['f0_over_time'], sig_prop['f0_range'])

L_num_frames = noisy_speech_stft.shape[-1]
N_num_samples = noisy_speech.shape[-1]
alphas_dirichlet = sce.SpectralCorrelationEstimator.get_alpha_vec_hz_dirichlet(L_num_frames,
                                                                               dft_props['r_shift_samples'],
                                                                               dft_props['fs'], sig_prop['f_max_hz'])
delta_f, delta_alpha_dict = m.compute_spectral_and_cyclic_resolutions(dft_props['fs'], dft_props['nfft'], ['acp'],
                                                                      alphas_dirichlet, N_num_samples)
delta_alpha = delta_alpha_dict['acp']
harmonic_bins = m.calculate_harmonic_frequencies(sig_prop['f0_range'],
                                                 sig_prop['num_harmonics'],
                                                 delta_alpha_dict['acp'],
                                                 sig_prop['alpha_max_hz'])
alpha_vec_hz = harmonic_bins * delta_alpha

speech_ir_mod = sce.SpectralCorrelationEstimator.modulate_signal_all_alphas_vec_2d(y=speech_ir,
                                                                                   alpha_vec_hz=alpha_vec_hz, fs_=fs)
speech_atf_mod = np.fft.fft(speech_ir_mod, n=dft_props['nfft'])
speech_rtf_mod = speech_atf_mod / speech_atf_mod[0, :]

# 2. Estimation of the spectral correlation function
# Create all the frequency shifted versions of the target signal
# wet_speech_mod = sce.SpectralCorrelationEstimator.modulate_signal_all_alphas_vec_2d(alpha_vec_hz=alpha_vec_hz, fs_=fs, y=wet_speech)
# wet_speech_stft_mod = local_stft(wet_speech_mod)

noisy_speech_mod = sce.SpectralCorrelationEstimator.modulate_signal_all_alphas_vec_2d(y=noisy_speech,
                                                                                      alpha_vec_hz=alpha_vec_hz, fs_=fs)
noisy_speech_stft_mod = local_stft(noisy_speech_mod)

# Compute the spatial/cyclic spectral correlation function for each spectral frequency
# First, test for a single frequency
f0_hz = sig_prop['f0_range'][1]

beamformed = np.zeros_like(noisy_speech_stft[0])
for kk in range(speech_atf.shape[-1]):

    kk_hz = kk * delta_f

    # Retain only shifts such that the cyclic frequency is less than the spectral frequency
    valid_shifts = np.where(alpha_vec_hz - kk_hz + f0_hz < 0)[0]

    # 0 should be included in the valid shifts
    if 0 not in valid_shifts:
        valid_shifts = np.concatenate(([0], valid_shifts))

    alpha_vec_hz_kk = alpha_vec_hz[valid_shifts]
    noisy_speech_stft_mod_kk = noisy_speech_stft_mod[:, valid_shifts, kk]

    noisy_speech_stft_mod_kk_2d = np.reshape(noisy_speech_stft_mod_kk, (-1, L_num_frames), order='F')

    # Compute covariance over dimension 0, then average over dimension 1
    cov_noisy_speech_stft_mod_kk = np.cov(noisy_speech_stft_mod_kk_2d, rowvar=True)
    cov_noisy_speech_stft_mod_kk += np.identity(cov_noisy_speech_stft_mod_kk.shape[0]) * 1e-8

    # u.plot_matrix(cov_noisy_speech_stft_mod_kk, title='Covariance matrix')

    speech_rtf_mod_kk = speech_rtf_mod[:, valid_shifts, kk]
    speech_rtf_mod_kk_2d = np.reshape(speech_rtf_mod_kk, (-1, 1), order='F')

    # we want to compute (cov_noisy_speech_stft_mod_kk)^-1 * atf_mod_kk implicitly
    # temp = np.linalg.solve(cov_noisy_speech_stft_mod_kk, speech_rtf_mod_kk_2d)
    temp = scipy.linalg.solve(cov_noisy_speech_stft_mod_kk, speech_rtf_mod_kk_2d, assume_a='pos')
    mvdr_beamformer = (temp / (np.conj(speech_rtf_mod_kk_2d.T) @ temp))

    # Apply the beamformer to the frequency domain signals
    beamformed[kk] = mvdr_beamformer.conj().T @ noisy_speech_stft_mod_kk_2d

noisy_time = u.normalize_volume(local_istft(noisy_speech_stft, real_data=True).real)
beamformed_time = u.normalize_volume(local_istft(beamformed, real_data=True).real)
u.plot([noisy_time, beamformed_time])
