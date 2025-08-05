"""
Cyclic beamforming in frequency domain, directly.
This script generates synthetic data and applies the MVDR beamformer to the data.
Useful for debugging and testing the beamformer.
"""
import copy
from pathlib import Path

import librosa
import numpy as np
import scipy

import pystoi


def quick_stoi(y):
    yy = u.normalize_and_pad(y, pad_to_len_=wet_target.shape[-1])
    ww = u.normalize_and_pad(wet_target[0], pad_to_len_=wet_target.shape[-1])
    return pystoi.stoi(x=ww, y=yy, fs_sig=fs, extended=False)


def compute_phase_correction_stft(stft_shape, overlap):
    """
    Compute the correction term to account for delay in STFT. See for example Equation 3 in
    "Fast computation of the spectral correlation" by Antoni, 2017.

    :param stft_shape: shape of the stft matrix, e.g. (num_mics, num, num_frames)
    :param overlap: a number between 0 and win_len-1, where win_len = (num_freqs_real-1)*2
    :return:
    """

    (_, num_freqs_real, num_frames) = stft_shape

    if num_frames == 1:
        return np.ones((num_freqs_real, 1))

    win_len = (num_freqs_real - 1) * 2  # delay correction depends on window size = win_len
    shift_samples = win_len - overlap

    # normalized frequencies in [0, 0.5] (real part)
    frequencies = np.arange(0, num_freqs_real)[:, np.newaxis] / win_len
    time_frames = np.arange(0, shift_samples * num_frames, shift_samples)[np.newaxis, :]
    correction_term = np.exp(-2j * np.pi * frequencies * time_frames)

    return correction_term


def compute_snr(original_, denoised_):
    """
    Compute the SNR in dB between the denoised and the original signal.
    """
    original = u.normalize_volume(original_)
    denoised = u.normalize_volume(denoised_)

    noise = original - denoised
    snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))
    return snr


def get_freq_shifted_signal(x_stft, integer_shifts_, max_bin_, kk_):
    shifted_bins_ = [kk_ - shift for shift in integer_shifts_]
    shifted_bins_ = [shift for shift in shifted_bins_ if 0 <= shift < max_bin_]

    if len(shifted_bins_) == 0:
        return x_stft[:, kk_:kk_ + 1, :]

    shifted_x = [x_stft[:, which_bin] for which_bin in shifted_bins_]
    shifted_x = np.stack(shifted_x, axis=1)

    np.allclose(x_stft[:, kk_, :], shifted_x[:, 0, :])

    return shifted_x


import globs

globals.rng = globals.compute_rng(seed_is_random=False, rnd_seed_=4110726324)
rng = globals.rng

import src.utils as u, manager
import modulator

u.set_printoptions_numpy()

fs = 16000
N = int(fs * 1)
M = 2  # Number of microphones

win_name = 'hann'
dft_props = {'nfft': 1024, 'fs': fs}
dft_props['nw'] = dft_props['nfft']
dft_props['noverlap'] = np.ceil(3 * dft_props['nw'] / 4).astype(int)
dft_props['r_shift_samples'] = dft_props['nw'] - dft_props['noverlap']
dft_props['delta_f'] = dft_props['fs'] / dft_props['nfft']
win = scipy.signal.windows.get_window(win_name, dft_props['nw'], fftbins=True)
window = win / np.sqrt(np.sum(win ** 2))  # normalize to unit power
nfft_real = dft_props['nfft'] // 2 + 1

if win_name == 'cosine' and dft_props['noverlap'] != dft_props['nw'] // 2:
    raise ValueError(f'For {win_name = }, {dft_props["noverlap"] = } must be {dft_props["nw"] // 2 = }')


def local_stft(y_, real_data=False):
    output_one_sided = real_data
    _, _, Y_ = scipy.signal.stft(y_, fs=fs, window=window, nperseg=dft_props['nw'], noverlap=dft_props['noverlap'],
                                 detrend=False, return_onesided=output_one_sided, boundary=None, padded=False, axis=-1)
    return Y_


def local_istft(Y_, real_data=False):
    input_one_sided = real_data
    _, y_ = scipy.signal.istft(Y_, fs=fs, window=window, nperseg=dft_props['nw'], noverlap=dft_props['noverlap'],
                               nfft=dft_props['nfft'], input_onesided=input_one_sided)
    return y_


m = manager.Manager()

# Paths
project_path = Path(__file__).resolve().parent.parent
datasets_path = project_path.parent / 'datasets'

# speech_dataset_path = datasets_path / 'north_texas_vowels' / 'data'
speech_dataset_path = datasets_path / 'audio'
speech_file_name = 'female.wav'
speech_path = speech_dataset_path / speech_file_name

ir_dataset_path = datasets_path / 'Hadad_shortcut'
speech_ir_path = ir_dataset_path / '1.wav'
dir_noise_ir_path = ir_dataset_path / '2.wav'

# Configs
rir_len_seconds = int(0.5 * fs)
snr_db_dir_noise = -10
snr_db_self_noise = 40
loading = 1e-8
minimize_noisy = True

# 1. Generation of synthetic data
# Load the speech sample
dry_target, target_fs = u.load_audio_file_random_offset(speech_path, fs=fs, duration_samples=N, smoothing_window=False)

# Load the impulse responses
speech_ir, speech_ir_fs = librosa.load(speech_ir_path, sr=fs, mono=False)
speech_ir = speech_ir[:M, :rir_len_seconds]  # Take only the first 0.5 seconds of the impulse response
normalization = np.sum(speech_ir ** 2)
speech_ir = speech_ir / normalization  # Normalize the impulse response to unit energy

dir_noise_ir, dir_noise_ir_fs = librosa.load(dir_noise_ir_path, sr=fs, mono=False)
dir_noise_ir = dir_noise_ir[:M, :rir_len_seconds]  # Take only the first 0.5 seconds of the impulse response
dir_noise_ir = dir_noise_ir / normalization  # Normalize the impulse response to unit energy

speech_atf = np.fft.rfft(speech_ir, n=dft_props['nfft'])

speech_atf = speech_atf + rng.normal(0, 5e-2, size=speech_atf.shape)  # Add noise to the ATF

speech_rtf = speech_atf / speech_atf[0, :]

# Check if the sampling rates are the same
if target_fs != speech_ir_fs:
    raise ValueError(f"Sampling rates are different: {target_fs = }, {speech_ir_fs = }")

# Convolve the speech with the impulse response
wet_target = scipy.signal.convolve(dry_target[np.newaxis, :], speech_ir, mode='full')[:, :len(dry_target)]

# Add noise to the signal. Two different realizations for the noise.
# Realization 1: used for the mix
# Directional noise
max_num_harmonics = 200
freqs_hz = np.array([122 * n for n in range(1, max_num_harmonics + 1)])
freqs_hz = freqs_hz[freqs_hz < fs / 2]
max_num_harmonics = len(freqs_hz)

phases = rng.uniform(-np.pi, np.pi, max_num_harmonics)
scaling_factors = rng.uniform(0.1, 100, max_num_harmonics)
dir_noise_dry_mix = u.generate_harmonic_process(freqs_hz, N, fs, phases=phases, local_rng=rng,
                                                scaling_factors=scaling_factors)
dir_noise_dry_mix = dir_noise_dry_mix[np.newaxis, :]

# dir_noise_dry_mix = rng.normal(0, 1, size=(1, dry_target.shape[-1]))
dir_noise_wet_mix = scipy.signal.convolve(dir_noise_dry_mix, dir_noise_ir, mode='full')[:, :len(dry_target)]
_, dir_noise_wet_mix = m.add_noise_snr(wet_target, snr_db=snr_db_dir_noise, fs=fs, noise=dir_noise_wet_mix)

# Self noise (microphone noise)
_, self_noise_wet_mix = m.add_noise_snr(wet_target, snr_db=snr_db_self_noise, fs=fs)

# Mix the signals
noisy_speech = wet_target + dir_noise_wet_mix + self_noise_wet_mix

# Realization 2: used for the covariance estimation
# Directional noise
# dir_noise_dry_for_cov = rng.normal(0, 1, size=(1, dry_target.shape[-1]))
phases = phases + rng.uniform(-np.pi / 2, np.pi / 2, size=1)
dir_noise_dry_for_cov = u.generate_harmonic_process(freqs_hz, N, fs, phases=phases, local_rng=rng,
                                                    scaling_factors=scaling_factors)
dir_noise_dry_for_cov = dir_noise_dry_for_cov[np.newaxis, :]
dir_noise_wet_for_cov = scipy.signal.convolve(dir_noise_dry_for_cov, dir_noise_ir, mode='full')[:, :len(dry_target)]
_, dir_noise_wet_for_cov = m.add_noise_snr(wet_target, snr_db=snr_db_dir_noise, fs=fs, noise=dir_noise_wet_for_cov)

# Self noise (microphone noise)
_, self_noise_wet_for_cov = m.add_noise_snr(wet_target, snr_db=snr_db_self_noise, fs=fs)

# Mix the noise signals (for the covariance estimation)
sum_noise_for_cov = dir_noise_wet_for_cov + self_noise_wet_for_cov

# Compute the frequency domain signals
wet_target_stft = local_stft(wet_target, real_data=True)
noisy_speech_stft = local_stft(noisy_speech, real_data=True)
sum_noise_for_cov_stft = local_stft(sum_noise_for_cov, real_data=True)

# Apply phase correction
phase_correction = compute_phase_correction_stft((1, nfft_real, noisy_speech_stft.shape[-1]), dft_props['noverlap'])
noisy_speech_stft_corr = noisy_speech_stft * phase_correction
sum_noise_for_cov_stft_corr = sum_noise_for_cov_stft * phase_correction[..., :sum_noise_for_cov_stft.shape[-1]]

L_num_frames = noisy_speech_stft.shape[-1]
N_num_samples = noisy_speech.shape[-1]

# 2. Narrowband beamforming
beamformed = np.zeros_like(noisy_speech_stft[0])
mvdr_beamformer = np.zeros((M, nfft_real), dtype=np.complex128)
for kk in range(speech_atf.shape[-1]):
    cov_noise_kk = (sum_noise_for_cov_stft_corr[:, kk] @ np.conj(sum_noise_for_cov_stft_corr[:, kk]).T) / L_num_frames
    cov_noisy_kk = (noisy_speech_stft_corr[:, kk] @ np.conj(noisy_speech_stft_corr[:, kk]).T) / L_num_frames

    # we want to compute (cov_noisy_speech_stft_mod_kk)^-1 * atf_mod_kk implicitly
    cov_kk = cov_noisy_kk if minimize_noisy else cov_noise_kk
    cov_inv_rtf = scipy.linalg.solve(cov_kk + loading * np.identity(M), speech_rtf[:, kk], assume_a='pos')
    mvdr_beamformer[:, kk] = cov_inv_rtf / (np.conj(speech_rtf[:, kk].T) @ cov_inv_rtf)

    # Apply the beamformer to the frequency domain signals
    beamformed[kk] = mvdr_beamformer[:, kk].conj().T @ noisy_speech_stft[:, kk]

noisy_time = u.normalize_volume(local_istft(noisy_speech_stft, real_data=True).real)
bfd_time = u.normalize_volume(local_istft(beamformed, real_data=True).real)

# 3. Cyclic beamforming
import f0_manager

f0_man = f0_manager.F0Manager()
all_freqs_hz = np.fft.fftfreq(dft_props['nfft'], 1 / dft_props['fs'])
harmonic_bins_shifts, _, _ = f0_man.find_harmonic_bins(fundamental_freq_hz=freqs_hz[0], freq_range_allowed=[0, 8000],
                                                       max_relative_dist_from_harmonic=0.5, all_freqs_hz=all_freqs_hz)
harmonic_bins_process, _, _ = f0_man.find_harmonic_bins(fundamental_freq_hz=freqs_hz[0], freq_range_allowed=[0, 8000],
                                                        max_relative_dist_from_harmonic=2.0, all_freqs_hz=all_freqs_hz)
P_max = 3
lcmv = False
integer_shifts_og = harmonic_bins_shifts[:P_max - 1]
# pre-append 0 to the integer shifts
integer_shifts_og = np.concatenate(([0], integer_shifts_og))
beamformed_cyclic = np.zeros_like(noisy_speech_stft[0])
beamformed_cyclic_clean = np.zeros_like(noisy_speech_stft[0])
cmvdr_beamformer = np.zeros((M * P_max, nfft_real), dtype=np.complex128)

# Make sure that if integer_shifts = [0], the signal is not shifted
# integer_shifts_test = [0]
# shifted_test = get_freq_shifted_signal(noisy_speech_stft_corr, integer_shifts_test, noisy_speech_stft_corr.shape[1], 2)
# assert np.allclose(noisy_speech_stft_corr[:, 2], shifted_test[:, 0])

for kk in range(speech_atf.shape[-1]):

    if kk in harmonic_bins_process:
        integer_shifts = integer_shifts_og.copy()
    else:
        integer_shifts = [0]

    max_bin = speech_atf.shape[-1] - 1
    shifted_noise = get_freq_shifted_signal(sum_noise_for_cov_stft_corr, integer_shifts, max_bin, kk)
    P = shifted_noise.shape[1]

    shifted_noisy = get_freq_shifted_signal(noisy_speech_stft_corr, integer_shifts, max_bin, kk)

    # Compute the spectral-spatial covariance matrix
    # Compute also the off-block-diagonal elements
    cov_kk_cyclic = np.zeros((M * P, M * P), dtype=np.complex128)
    for pp in range(P):
        for qq in range(P):
            if minimize_noisy:
                cov_kk_cyclic[pp * M:(pp + 1) * M, qq * M:(qq + 1) * M] = (shifted_noisy[:, pp] @ np.conj(shifted_noisy[:, qq]).T)
            else:
                cov_kk_cyclic[pp * M:(pp + 1) * M, qq * M:(qq + 1) * M] = (shifted_noise[:, pp] @ np.conj(shifted_noise[:, qq]).T)
    cov_kk_cyclic = cov_kk_cyclic / L_num_frames

    if lcmv:
        # C_rtf contains in the first column the RTF padded with zeroes. In the second column, zeros, then the RTF
        # at the first harmonic, and so on.
        shifted_bins = [kk + shift for shift in integer_shifts]
        shifted_bins = [shift for shift in shifted_bins if 0 <= shift < max_bin]
        C_rtf = np.zeros((M * P, P), dtype=np.complex128)
        if shifted_bins:
            for pp in range(P):
                try:
                    C_rtf[pp * M:(pp + 1) * M, pp] = speech_rtf[:, shifted_bins[pp]]
                except IndexError:
                    C_rtf[:M, 0] = speech_rtf[:, kk]
        else:
            C_rtf[:M, 0] = speech_rtf[:, kk]

        const = np.zeros(P)
        const[0] = 1
        cov_wb_kk_inv_c = scipy.linalg.solve(cov_kk_cyclic + 1e-8 * np.identity(M * P), C_rtf, assume_a='pos')
        if P == 1:
            cmvdr_beamformer[:M * P, kk] = np.squeeze(
                cov_wb_kk_inv_c / (C_rtf.conj().T @ cov_wb_kk_inv_c).real)  # just MVDR
        else:
            try:
                cmvdr_beamformer[:M * P, kk] = cov_wb_kk_inv_c @ np.linalg.inv(C_rtf.conj().T @ cov_wb_kk_inv_c) @ const
            except np.linalg.LinAlgError:
                print(f"LinAlgError for bin {kk}")
    else:
        # rtf padded with 0 to match the size of the covariance matrix
        speech_rtf_padded = np.zeros((M * P), dtype=np.complex128)
        speech_rtf_padded[:M] = speech_rtf[:, kk]

        # we want to compute (cov_noisy_speech_stft_mod_kk)^-1 * atf_mod_kk implicitly
        cov_cyclic_inv_rtf = scipy.linalg.solve(cov_kk_cyclic + loading * np.identity(M * P), speech_rtf_padded,
                                                assume_a='pos')
        cmvdr_beamformer[:M * P, kk] = cov_cyclic_inv_rtf / (np.conj(speech_rtf_padded.T) @ cov_cyclic_inv_rtf)

    assert P > 0

    # Apply the beamformer to the frequency domain signals
    shifted_noisy_output = get_freq_shifted_signal(noisy_speech_stft, integer_shifts, max_bin, kk)
    shifted_noisy_output_2d = np.reshape(shifted_noisy_output, (M * P, -1), order='F')
    beamformed_cyclic[kk] = cmvdr_beamformer[:M * P, kk].conj().T @ shifted_noisy_output_2d

    shifted_wet = get_freq_shifted_signal(wet_target_stft, integer_shifts, max_bin, kk)
    shifted_wet_2d = np.reshape(shifted_wet, (M * P, -1), order='F')
    beamformed_cyclic_clean[kk] = cmvdr_beamformer[:M * P, kk].conj().T @ shifted_wet_2d

bfd_cyclic_time = local_istft(beamformed_cyclic, real_data=True).real
bfd_cyclic_time = u.normalize_volume(bfd_cyclic_time)

bfd_cyclic_clean_time = local_istft(beamformed_cyclic_clean, real_data=True).real
bfd_cyclic_clean_time = u.normalize_volume(bfd_cyclic_clean_time)

# u.plot([noisy_time, beamformed_time, bfd_cyclic], fs, ['Noisy', 'MVDR', 'Cyclic MVDR'], subplot_height=1.5)

# Calculate the STOI
stoi_noisy = quick_stoi(noisy_time[0])
stoi_bfd = quick_stoi(bfd_time)
stoi_bfd_cyclic = quick_stoi(bfd_cyclic_time)
stoi_bfd_cyclic_clean = quick_stoi(bfd_cyclic_clean_time)

print(f"STOI Noisy: {stoi_noisy:.3f}, MVDR: {stoi_bfd:.3f}, Cyclic MVDR: {stoi_bfd_cyclic:.3f}, "
      f"Cyclic MVDR Clean: {stoi_bfd_cyclic_clean:.3f}")

# plot_spectrograms(signals_dict, freqs_hz, max_displayed_frequency_bin=max_displayed_frequency_bin,
#                           max_time_frames=max_time_frames, save_figs=plot_settings['save_plots'],
#                           slice_chunks=slice_bf_list, delta_t=delta_t, suptitle=debug_title)
import plotter as pl

# pl.plot_spectrograms({'Noisy': noisy_speech_stft[0], 'MVDR': beamformed, 'Cyclic MVDR': beamformed_cyclic,
#                       'Cyclic MVDR Clean': beamformed_cyclic_clean}, all_freqs_hz)
pl.plot_spectrograms({'MVDR': beamformed, 'Cyclic MVDR': beamformed_cyclic}, all_freqs_hz)

if 0:
    u.play(noisy_time, fs, volume=0.5)
    u.play(bfd_time, fs, volume=0.5)
    u.play(bfd_cyclic_time, fs, volume=0.5)
    u.play(bfd_cyclic_clean_time, fs, volume=0.5)
    u.play(wet_target[0], fs, volume=0.5)
