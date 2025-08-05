import numpy as np
from copy import deepcopy as dcopy

from scipy.signal import ShortTimeFFT

from src import globs as gs, config
from src.data_generator import DataGenerator

cfg_original = config.load_configuration('default.yaml')
cfg_original = config.load_and_merge_secondary_config(cfg_original)
gs.rng, cfg_original['seed_extracted'] = gs.compute_rng(cfg_original['seed_is_random'],
                                                        cfg_original['seed_if_not_random'])

from src.beamformer_manager import BeamformerManager
from src import (utils as u, covariance_estimator,
                                   f0_manager, manager, plotter as pl)
from src.f0_manager import F0ChangeAmount
from src.harmonic_info import HarmonicInfo
from src.modulator import Modulator
from src.coherence_manager import CoherenceManager
from src.player import Player  # do not remove, useful for quick evaluation of signals from cmd line

u.set_printoptions_numpy()
threshold_hz_f0_std = np.inf


class ExperimentManager:
    def __init__(self):
        pass

    @staticmethod
    def run_cov_estimation_beamforming(signals, f0man, f0_over_time, harmonic_freqs_est, cfg, dft_props, do_plots,
                                       SFT: ShortTimeFFT, name_input_sig='noisy'):

        def print_log_chunks():
            if do_plots:
                slice_cov_est = slice_cov_est_list[idx_chunk]
                if num_chunks > 1:
                    print(f"Process chunk by chunk. Chunk {idx_chunk + 1}/{num_chunks}, "
                          f"frames {slice_cov_est.start}-{slice_cov_est.stop}/{L1_frames_all_chunks}.")
                else:
                    print(f"Process all chunks at once. {num_chunks = }, {L1_frames_all_chunks = }, signal duration = "
                          f"{L1_frames_all_chunks * SFT.delta_t:.2f}s.")

        def print_log_noise():
            if do_plots:
                print(f"Noise signals generated. "
                      f"Total duration is {int(cfg['cov_estimation']['noise_cov_est_len_seconds'] * SFT.fs) / SFT.fs:.2f}s.")

        # cov_dict_prev = covariance_estimator.CovarianceHolder()  # empty covariance holder
        cov_dict_prev = {}
        coh_man = CoherenceManager()
        m = manager.Manager()
        f0_tracker = f0_manager.F0Tracker()

        K_nfft_real = dft_props['nfft_real']
        is_cmwf_bf = cfg['cyclostationary_target']
        use_pseudo_cov = any(['wl' in x for x in cfg['beamforming']['methods']])
        ce = covariance_estimator.CovarianceEstimator(cfg['cov_estimation'], cfg['cyclostationary_target'],
                                                      subtract_mean=False, use_pseudo_cov=use_pseudo_cov)
        M = signals[name_input_sig]['stft'].shape[0]
        # use_masked_stft_for_evaluation = cfg['use_masked_stft_for_evaluation']

        target_rtf = np.array([])
        if 'wet' in signals:
            target_rtf = DataGenerator.calculate_ground_truth_rtf(signals['wet'])

        print_log_noise()

        # if np.isfinite(cfg['est_error_snr']):
        #     signals_unproc_with_est_noise = m.add_noise_to_signals(dcopy(signals_unproc),
        #                                                            cfg['est_error_snr'],
        #                                                            which_signals=['wet_rank1'])

        L1_frames_all_chunks = signals[name_input_sig]['stft'].shape[-1]
        bfd_all_chunks_stft, bfd_all_chunks_stft_masked = m.allocate_beamformed_signals(
            L1_frames_all_chunks, K_nfft_real, cfg['beamforming']['methods'])

        slice_bf_list, slice_cov_est_list, num_chunks = \
            (m.get_chunks_slices(L1_frames_all_chunks, dft_props=dft_props, time_props=cfg['time'],
                                 recursive_average=cfg['cov_estimation']['recursive_average']))

        bf = BeamformerManager(beamformers_names=cfg['beamforming']['methods'],
                               sig_shape_k_m=(K_nfft_real, M),
                               minimize_noisy_cov_mvdr=cfg['beamforming']['minimize_noisy_cov_mvdr'],
                               loadings=cfg['beamforming']['loadings'],
                               noise_var_rtf=cfg['noise']['noise_var_rtf'])
        mod_amount_numeric_list = []
        harm_info = HarmonicInfo()
        num_voiced_chunks = 0
        cyclic_bins_ratio_list = []

        # Loop over the chunks of the signal to estimate the covariance matrices and beamform the signals
        for idx_chunk in range(num_chunks):
            if not bf.beamformers_names:
                continue

            is_first_chunk = idx_chunk == 0
            slice_bf = slice_bf_list[idx_chunk]
            print_log_chunks()

            if 'f0' in cfg['harmonics_est']['algo']:
                harmonic_freqs_est = f0man.get_harmonics_as_multiples_of_f0(f0_over_time[slice_bf],
                                                                            cfg['cyclic']['freq_range_cyclic'])

            mod_amount = F0ChangeAmount.no_change
            if is_cmwf_bf or is_first_chunk:
                harm_info, mod_amount = f0man.compute_harmonic_and_modulation_sets_distance_based(
                    harmonic_freqs_est, cfg['harmonics_est'], cfg['cyclic'], dft_props, idx_chunk,
                    f0_over_time[slice_bf], f0_tracker)

                if idx_chunk == 0:
                    print(f"Num shifts per set: {harm_info.num_shifts_per_set}")

                if harm_info.alpha_mods_sets[0].size > 1:
                    num_voiced_chunks = num_voiced_chunks + 1
                    cyclic_bins_ratio = (np.sum(harm_info.mask_harmonic_bins) /
                                         max(1, harm_info.mask_harmonic_bins.size))
                    cyclic_bins_ratio_list.append(cyclic_bins_ratio)

                ce.harmonic_info = harm_info
                bf.harmonic_info = harm_info

            signals_to_modulate = config.ConfigManager.choose_signals_to_modulate(
                is_cmwf_bf, cfg['beamforming']['minimize_noisy_cov_mvdr'],
                is_first_chunk, cfg['cov_estimation']['recursive_average'], name_input_sig)

            # Re-modulate the signals with new harmonic info
            if mod_amount == F0ChangeAmount.small:
                signals = Modulator.modulate_signals(signals, signals_to_modulate, SFT,
                                                     harm_info.alpha_mods_sets, cfg['cyclic']['P_max'],
                                                     name_input_sig)

            ce.set_dimensions((K_nfft_real, M, cfg['cyclic']['P_max']))
            cov_dict = ce.estimate_covariances(slice_cov_est_list[idx_chunk], signals, cov_dict_prev,
                                               num_mics_changed=ce.sig_shape_k_m_p[1] != M,
                                               modulation_amount=mod_amount,
                                               name_input_sig=name_input_sig)
            cov_dict_prev = dcopy(cov_dict)

            # Compute weights for beamformers
            weights, error_flags = bf.compute_weights_all_beamformers(cov_dict=cov_dict, rtf_oracle=target_rtf,
                                                                      idx_chunk=idx_chunk,
                                                                      name_input_sig=name_input_sig)

            # Beamform the signals
            mod_amount_bf = mod_amount if idx_chunk != 0 else F0ChangeAmount.no_change
            bfd = bf.beamform_signals(signals[name_input_sig]['stft'],
                                      signals[name_input_sig]['mod_stft_3d'], slice_bf, weights,
                                      mod_amount=mod_amount_bf)
            mod_amount_numeric_list.append(mod_amount.to_number())

            # Store the beamformed signals for the current chunk in the 'all_chunks' dictionary
            for key in bfd.keys():
                if key != name_input_sig:
                    bfd_all_chunks_stft[key][:, slice_bf] = bfd[key]

            # end of chunks loop

        bf.check_beamformed_signals_non_zero(bfd_all_chunks_stft, signals)

        # Plot waveforms and spectrograms
        if do_plots:
            bench = 'mwf_blind' if is_cmwf_bf else 'mvdr_blind'
            # debug_title = f"{parameter_to_vary} = {str(param_value)}",
            pl.plot_waveforms_and_spectrograms(dcopy(cfg['plot']), bfd_all_chunks_stft, dft_props,
                                               f0_tracker.alpha_smooth_list,
                                               num_chunks, slice_bf_list, dft_props['nfft'], SFT.delta_t,
                                               freq_max_cyclic=cfg['cyclic']['freq_range_cyclic'][1],
                                               benchmark_algo=bench)

        return bfd_all_chunks_stft

    @staticmethod
    def convert_signals_time_domain(bfd_all_chunks_stft, SFT_real):
        # Compute the time-domain signals from the beamformed STFTs and append to dict
        signals_bfd_dict = {}
        for key in bfd_all_chunks_stft.keys():
            signals_bfd_dict[key] = {'time': SFT_real.istft(bfd_all_chunks_stft[key]).real,
                                     'stft': bfd_all_chunks_stft[key],
                                     # 'stft_masked': bfd_all_chunks_stft_masked[key],
                                     'display_name': pl.get_display_name(key)}

        return signals_bfd_dict
