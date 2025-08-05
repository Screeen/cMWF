"""
This script generates synthetic data and applies the MVDR beamformer to the data.
Useful for debugging and testing the beamformer.
"""

from pathlib import Path

import librosa
import numpy as np
import scipy
from src.globs import rng

from src import manager, utils as u

# Add the path to the cyclic beamforming package
module_parent = Path(__file__).resolve().parent

u.set_printoptions_numpy()

fs = 16000
N = -1
M = 8  # Number of microphones

win_name = 'hann'
dft_props = {'nfft': 512, 'fs': fs}
dft_props['nw'] = dft_props['nfft']
dft_props['noverlap'] = np.ceil(2 * dft_props['nw'] / 4).astype(int)
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

speech_dataset_path = datasets_path / 'north_texas_vowels' / 'data'
speech_file_name = 'kadpal08.wav'
speech_path = speech_dataset_path / speech_file_name

ir_dataset_path = datasets_path / 'Hadad_shortcut'
speech_ir_path = ir_dataset_path / '1.wav'
dir_noise_ir_path = ir_dataset_path / '2.wav'

# Configs
rir_len_seconds = int(0.75 * fs)
snr_db_dir_noise = -5
snr_db_self_noise = 40


# 1. Generation of synthetic data
# Load the speech sample
dry_speech, dry_speech_fs = u.load_audio_file(speech_path, fs_=fs, N_num_samples=N, offset_seconds=0.045,
                                              smoothing_window=False)

# Load the impulse responses
speech_ir, speech_ir_fs = librosa.load(speech_ir_path, sr=fs, mono=False)
speech_ir = speech_ir[:M, :rir_len_seconds]  # Take only the first 0.5 seconds of the impulse response
normalization = np.sum(speech_ir ** 2)
speech_ir = speech_ir / normalization  # Normalize the impulse response to unit energy

dir_noise_ir, dir_noise_ir_fs = librosa.load(dir_noise_ir_path, sr=fs, mono=False)
dir_noise_ir = dir_noise_ir[:M, :rir_len_seconds]  # Take only the first 0.5 seconds of the impulse response
dir_noise_ir = dir_noise_ir / normalization  # Normalize the impulse response to unit energy

speech_atf = np.fft.rfft(speech_ir, n=dft_props['nfft'])
speech_rtf = speech_atf / speech_atf[0, :]

# Check if the sampling rates are the same
if dry_speech_fs != speech_ir_fs:
    raise ValueError(f"Sampling rates are different: {dry_speech_fs = }, {speech_ir_fs = }")

# Convolve the speech with the impulse response
wet_speech = scipy.signal.convolve(dry_speech[np.newaxis, :], speech_ir, mode='full')[:, :len(dry_speech)]

# Add noise to the signal. Two different realizations for the noise.
# Realization 1: used for the mix
# Directional noise
dir_noise_dry_mix = rng.normal(0, 1, size=(1, dry_speech.shape[-1]))
dir_noise_wet_mix = scipy.signal.convolve(dir_noise_dry_mix, dir_noise_ir, mode='full')[:, :len(dry_speech)]
_, dir_noise_wet_mix = m.add_noise_snr(wet_speech, snr_db=snr_db_dir_noise, fs=fs, noise=dir_noise_wet_mix)

# Self noise (microphone noise)
_, self_noise_wet_mix = m.add_noise_snr(wet_speech, snr_db=snr_db_self_noise, fs=fs)

# Mix the signals
noisy_speech = wet_speech + dir_noise_wet_mix + self_noise_wet_mix

# Realization 2: used for the covariance estimation
# Directional noise
dir_noise_dry_for_cov = rng.normal(0, 1, size=(1, dry_speech.shape[-1]))
dir_noise_wet_for_cov = scipy.signal.convolve(dir_noise_dry_for_cov, dir_noise_ir, mode='full')[:, :len(dry_speech)]
_, dir_noise_wet_for_cov = m.add_noise_snr(wet_speech, snr_db=snr_db_dir_noise, fs=fs, noise=dir_noise_wet_for_cov)

# Self noise (microphone noise)
_, self_noise_wet_for_cov = m.add_noise_snr(wet_speech, snr_db=snr_db_self_noise, fs=fs)

# Mix the noise signals (for the covariance estimation)
sum_noise_for_cov = dir_noise_wet_for_cov + self_noise_wet_for_cov

# Compute the frequency domain signals
noisy_speech_stft = local_stft(noisy_speech, real_data=True)
sum_noise_for_cov_stft = local_stft(sum_noise_for_cov, real_data=True)

L_num_frames = noisy_speech_stft.shape[-1]
N_num_samples = noisy_speech.shape[-1]

# 2. Beamforming
beamformed = np.zeros_like(noisy_speech_stft[0])
mvdr_beamformer = np.zeros((M, nfft_real), dtype=np.complex128)
for kk in range(speech_atf.shape[-1]):
    cov_noise_kk = (sum_noise_for_cov_stft[:, kk] @ np.conj(sum_noise_for_cov_stft[:, kk]).T) / L_num_frames

    # we want to compute (cov_noisy_speech_stft_mod_kk)^-1 * atf_mod_kk implicitly
    cov_noise_inv_rtf = scipy.linalg.solve(cov_noise_kk, speech_rtf[:, kk], assume_a='pos')
    mvdr_beamformer[:, kk] = cov_noise_inv_rtf / (np.conj(speech_rtf[:, kk].T) @ cov_noise_inv_rtf)

    # Apply the beamformer to the frequency domain signals
    beamformed[kk] = mvdr_beamformer[:, kk].conj().T @ noisy_speech_stft[:, kk]

noisy_time = u.normalize_volume(local_istft(noisy_speech_stft, real_data=True).real)
beamformed_time = u.normalize_volume(local_istft(beamformed, real_data=True).real)
u.plot([noisy_time[0], beamformed_time], titles=['Noisy', 'Beamformed'], fs=fs)

#
u.play(noisy_time[0], fs)
u.play(beamformed_time, fs)
