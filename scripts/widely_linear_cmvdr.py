import copy

import numpy as np
import scipy

from src import globs as gs

gs.rng, _ = gs.compute_rng(seed_is_random=0, rnd_seed_=15)
from src.covariance_estimator import CovarianceEstimator, CovarianceHolder

from src.beamformer_manager import BeamformerManager
from src import (utils as u, data_generator, evaluator,
                                   f0_manager, manager, plotter as pl)
from src.f0_manager import F0ChangeAmount
from src.harmonic_info import HarmonicInfo
from src.modulator import Modulator
from src.sin_generator import SinGenerator

"""
This script is to test the potential usefulness of widely-linear cyclic beamforming.
"""


def compute_wl_mvdr_beamformers(ch_, processed_bins, speech_rtf, P_all_, loadings, loadings_nb):

    """ Compute the weights for the WIDELY LINEAR cyclic MVDR beamformer (cMVDR). """

    K_nfft_, M_ = ch_.noisy_nb.shape[:2]
    P_max_ = ch_.noisy_wb.shape[-1] // M_
    weights_ = np.zeros((M_ * P_max_, K_nfft_), dtype=np.complex128, order='F')

    eye_nb = np.eye(M_)
    eye_wb = np.eye(M_ * P_max_)

    for k in range(K_nfft_):
        P_ = P_all_[k] if P_all_.size > 0 else 1
        rtf = speech_rtf[k]
        if k not in processed_bins:  # Fall back to MVDR for the non-harmonic bins
            cov_kk = ch_.noisy_nb[k]
            cov_nb_kk_inv_rtf = scipy.linalg.solve(cov_kk + loadings_nb[k] * eye_nb, rtf, assume_a='pos')
            weights_[:M_, k] = np.squeeze(cov_nb_kk_inv_rtf / (np.conj(rtf).T @ cov_nb_kk_inv_rtf).real)

        else:  # For the harmonic bins, use the cMVDR
            cov_kk = ch_.noisy_wb[k]
            d2 = M_ * (P_ // 2) - M_
            rtf_padded = np.concatenate((rtf, np.zeros(d2), np.conj(rtf), np.zeros(d2)))
            cov_wb_kk_inv_rtf = scipy.linalg.solve(
                cov_kk[:M_ * P_, :M_ * P_] + loadings[k] * eye_wb[:M_ * P_, :M_ * P_],
                rtf_padded, assume_a='pos')

            weights_[:M_ * P_, k] = cov_wb_kk_inv_rtf / (np.conj(rtf_padded).T @ cov_wb_kk_inv_rtf).real

    return weights_


def beamform_signals_wl(noisy_stft, noisy_mod_stft_3d, slice_frames_, wl_weights, P_all_):

    # if we are towards the end of the signal, the actual number of frames might be less than the expected
    # noisy_stft is always provided in full
    L3_num_frames = min(slice_frames_.stop - slice_frames_.start, noisy_stft.shape[-1] - slice_frames_.start)
    K_nfft_real_ = noisy_stft.shape[1]

    bfd_stft = np.zeros((K_nfft_real_, L3_num_frames), dtype=np.complex128, order='F')
    M_ = noisy_stft.shape[0]

    for kk__ in range(K_nfft_real_):
        # Determine the harmonic set and the number of shifts per set
        P_ = P_all_[kk__]
        sel_weights = slice(M_ * P_), kk__
        sel_signal = 0, slice(M_ * P_), kk__, slice_frames_
        bfd_stft[kk__] = np.conj(wl_weights[sel_weights]).T @ noisy_mod_stft_3d[sel_signal]

    return bfd_stft


u.set_printoptions_numpy()
dg = data_generator.DataGenerator()
sin_gen_target = SinGenerator(correlation_factor=0.2)
sin_gen_noise = SinGenerator(correlation_factor=0.2)
# sin_gen_noise.scaling_factors = np.ones_like(sin_gen_noise.scaling_factors)
f0_man = f0_manager.F0Manager()
fs = 16000
duration_samples = int(fs * 1.)
# f0_hz = 523.8  # trumpet
f0_hz = 780.27  # violin
# f0_hz = 22.3 * 2  # keykrusher
P_max = 10
P_model = P_max + 1
K_nfft = 1024

window = scipy.signal.windows.get_window('hann', K_nfft, fftbins=True)
SFT = scipy.signal.ShortTimeFFT(hop=K_nfft // 4, fs=fs, win=window, fft_mode='twosided', scale_to='magnitude')
SFT_real = scipy.signal.ShortTimeFFT(hop=K_nfft // 4, fs=fs, win=window, fft_mode='onesided', scale_to='magnitude')

# Generate the signals
# target, _ = dg.generate_or_load_anechoic_signal(duration_samples, fs, sig_type='white')
target, _ = dg.generate_or_load_anechoic_signal(duration_samples, fs, sig_type='sample',
                                                sample_path=r'/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/audio/400a0j0a.wav')
                                                # sample_path=r'/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/audio/wideroad.wav')
# target2, _ = dg.generate_or_load_anechoic_signal(duration_samples, fs, sig_type='sinusoidal', f0_hz=139,
#                                                     sin_gen=sin_gen_target, num_harmonics=P_model)
# target = 0.5 * target + 0.5 * target2

# noise_harm, _ = dg.generate_or_load_anechoic_signal(duration_samples, fs, sig_type='sinusoidal', f0_hz=f0_hz,
#                                                     sin_gen=sin_gen_noise, num_harmonics=P_model)
noise_harm, _ = dg.generate_or_load_anechoic_signal(duration_samples, fs, sig_type='sample',
                                                # sample_path=r'/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/audio/instruments-single-notes/trumpet-C5.wav')
                                                sample_path=r'/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/audio/instruments-single-notes/violin-G5.wav')
                                                # sample_path=r'/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/audio/160545__keykrusher__lathe-atlas-6inch-idling.wav')
noise_white, _ = dg.generate_or_load_anechoic_signal(duration_samples, fs, sig_type='white')
noise_white = noise_white * 5.e-3
noise_mix = noise_harm + noise_white
noise_mix = noise_mix * 0.9

target_f = SFT.stft(target)
atf_target = np.r_[1., 0.8 - 0.2j, 0.8 - 0.2j]
target_f = target_f * atf_target[:, np.newaxis, np.newaxis]
noise_f = SFT.stft(noise_mix)
atf_noise = np.r_[0.9 + 0.2j, 0.3 - 0.9, 0.3 - 0.9]
noise_f = noise_f * atf_noise[:, np.newaxis, np.newaxis]

noisy_mix_f = target_f + noise_f
noisy_mix = SFT.istft(noisy_mix_f).real
noisy_mix_f = SFT.stft(noisy_mix)
noisy_mix_f_conj = np.conj(noisy_mix_f)
M, K_nfft, L2_frames_chunk = noisy_mix_f.shape
K_nfft_real = K_nfft // 2 + 1
slice_frames = slice(0, noisy_mix_f_conj.shape[-1])

# Modulate signals
signals = {
    'noisy': {'stft': noisy_mix_f[:, :K_nfft_real], 'stft_conj': noisy_mix_f_conj[:, :K_nfft_real], 'time': noisy_mix},
    'noise': {'stft': noise_f[:, :K_nfft_real], 'stft_conj': np.conj(noise_f)[:, :K_nfft_real], 'time': noise_mix},
}
harmonic_freqs = np.asarray([f0_hz * n for n in np.arange(1, P_model)])
alpha_mods_sets = [np.concatenate((np.r_[0], -harmonic_freqs[:-2], harmonic_freqs[0:1]))]
signals = Modulator.modulate_signals(signals, ['noisy', 'noise'], SFT, alpha_mods_sets, P_max)

signals['noisy_wl'] = {}
signals['noisy_wl']['mod_stft_3d'] = np.concatenate(
    (signals['noisy']['mod_stft_3d'], signals['noisy']['mod_stft_3d_conj']), axis=1)
signals['noisy_wl']['mod_stft_3d_conj'] = np.conj(signals['noisy_wl']['mod_stft_3d'])

# Allocate covariance matrices
cov_dict = {
    'noisy_wl': np.zeros((K_nfft_real, 2 * M * P_max, 2 * M * P_max), dtype=np.complex128, order='F'),
    'noisy_wb': np.zeros((K_nfft_real, M * P_max, M * P_max), dtype=np.complex128, order='F'),
    'noise_wb': np.zeros((K_nfft_real, M * P_max, M * P_max), dtype=np.complex128, order='F'),
}

# Estimate covariance matrices
for kk_ in range(K_nfft_real):
    # Select the correct slice for the current frequency, depending on the harmonic set
    P = P_max
    sel = 0, slice(M * P), kk_, slice_frames
    sel_wl = 0, slice(2 * M * P), kk_, slice_frames
    sel_cov = kk_, slice(M * P), slice(M * P)  # corresponds to [kk, :M*P, :M*P]
    sel_cov_wl = kk_, slice(2 * M * P), slice(2 * M * P)  # corresponds to [kk, :M*P, :M*P]

    mod_stft = 'mod_stft_3d'
    mod_stft_conj = 'mod_stft_3d_conj'
    noisy_ = signals['noisy']
    noisy_wl = signals['noisy_wl']
    noise_ = signals['noise']
    cov_dict['noisy_wb'][sel_cov] = ((noisy_[mod_stft][sel] @ noisy_[mod_stft_conj][sel].T) / L2_frames_chunk)
    cov_dict['noise_wb'][sel_cov] = ((noise_[mod_stft][sel] @ noise_[mod_stft_conj][sel].T) / L2_frames_chunk)

    cov_dict['noisy_wl'][sel_cov_wl] = ((noisy_wl[mod_stft][sel_wl] @ noisy_wl[mod_stft_conj][sel_wl].T) / L2_frames_chunk)

noisy_wl_2 = np.zeros((K_nfft_real, 2 * M * P_max, 2 * M * P_max), dtype=complex)
P = P_max
for kk_ in range(K_nfft_real):
    sel_wb = 0, slice(M * P), kk_, slice_frames
    sel_cov_nw = kk_, slice(M * P), slice(M * P)  # corresponds to [kk, :M*P, :M*P]
    sel_cov_se = kk_, slice(M * P, 2 * M * P), slice(M * P, 2 * M * P)
    sel_cov_ne = kk_, slice(M * P), slice(M * P, 2 * M * P)
    sel_cov_sw = kk_, slice(M * P, 2 * M * P), slice(M * P)

    noisy_wl = signals['noisy_wl']
    mod_stft = 'mod_stft_3d'
    mod_stft_conj = 'mod_stft_3d_conj'

    # North-west
    noisy_wl_2[sel_cov_nw] = (
                (noisy_wl[mod_stft][sel_wb] @ noisy_wl[mod_stft_conj][sel_wb].T) / L2_frames_chunk)

    # South-east
    noisy_wl_2[sel_cov_se] = np.conj(noisy_wl_2[sel_cov_nw])

    # North-east
    noisy_wl_2[sel_cov_ne] = (
            (noisy_wl[mod_stft][sel_wb] @ noisy_wl[mod_stft][sel_wb].T) / L2_frames_chunk)

    # South-west
    noisy_wl_2[sel_cov_sw] = np.conj(noisy_wl_2[sel_cov_ne])

assert np.allclose(noisy_wl_2, cov_dict['noisy_wl'])

cov_dict = CovarianceEstimator.copy_multiband_to_narrowband(cov_dict, M=M)
cov_dict1 = copy.deepcopy(cov_dict)
cov_dict1.pop('noisy_wl')
ch = CovarianceHolder(**cov_dict1)

kk1 = int(np.round(f0_hz / SFT.delta_f))

# Harmonic bins are close to harmonic frequencies in "harmonic_freqs_est"
# Harmonic sets are collections of neighbouring harmonic bins
harmonic_bins, harmonic_sets = f0_man.calculate_harmonic_sets(harmonic_freqs, 2.5,
                                                              (0, 5000), 'f0_oracle', K_nfft, fs)

# Compute weights for beamformers
bf = BeamformerManager(beamformers_names=['mvdr_blind', 'cmvdr_blind'],
                       sig_shape_k_m=(K_nfft_real, M),
                       minimize_noisy_cov_mvdr=True,
                       noise_var_rtf=0,
                       loadings={'mvdr': (0, 0, 1000)})
bf.harmonic_info = HarmonicInfo(harmonic_bins=harmonic_bins, harmonic_sets=harmonic_sets, harmonic_freqs_hz=[f0_hz],
                                alpha_mods_sets=alpha_mods_sets)
weights, error_flags = bf.compute_weights_all_beamformers(ch=ch)

cov_dict['noisy_wb'] = copy.deepcopy(cov_dict['noisy_wl'])
cov_dict.pop('noisy_wl')
ch_wl = CovarianceHolder(**cov_dict)
P_all = bf.harmonic_info.get_num_shifts_all_frequencies()
P_all[P_all > 1] = 2*P_all[P_all > 1]
ww = compute_wl_mvdr_beamformers(ch_wl, bf.harmonic_info.harmonic_bins, bf.mvdr.rtf_est, P_all,
                                 bf.mvdr.get_loading_wb(ch_wl.noisy_wb, *bf.mvdr.loadings_cfg, P_all),
                                 bf.mvdr.get_loading_nb('blind', ch_wl.noisy_nb, *bf.mvdr.loadings_cfg))
bfd_wl = beamform_signals_wl(signals['noisy']['stft'], signals['noisy_wl']['mod_stft_3d'], slice_frames, ww, P_all)

# Beamform the signals
bfd = bf.beamform_signals(signals['noisy']['stft'],
                          signals['noisy']['mod_stft_3d'], slice_frames, weights,
                          mod_amount=F0ChangeAmount.small)

bfd['cmvdr_wl_blind'] = bfd_wl
bfd['noisy'] = noisy_mix_f[0, :K_nfft_real]

# Evaluate the signals
bf_time = []
max_val = 0
for bf_name in bfd.keys():
    mse_kk1 = np.mean(np.abs(bfd[bf_name][..., harmonic_bins, :target_f.shape[-1]] - target_f[0, harmonic_bins]) ** 2)
    mse_kk1_db = evaluator.to_db(mse_kk1)
    print(f"{bf_name = }, {mse_kk1_db = :.1f}dB")
    bf_time.append(SFT_real.istft(bfd[bf_name]))
    max_val = max(max_val, np.max(np.abs(bfd[bf_name])))

    target_pad = u.pad_last_dim(target, bf_time[-1].shape[-1])
    si_sdr_ = evaluator.Evaluator.si_sdr(target_pad, bf_time[-1])[0]
    print(f"\t\t\t\t\t\t\t\t\t\t {si_sdr_ = :.1f}dB")

for bf_name in bfd.keys():
    u.plot_matrix(evaluator.to_db(np.abs(bfd[bf_name]) / max_val), title=bf_name, log=False,
                  normalized=False, amp_range=(-60, 0))

if 0:
    u.plot(bf_time, titles=list(bfd.keys()), subplot_height=1.2)
    u.play(bf_time[3], volume=0.4)  # noisy
    u.play(bf_time[1], volume=0.4)  # cmvdr
    u.play(bf_time[2], volume=0.4)  # cmvdr WL

pass
