import copy
import warnings
from typing import Optional

import numpy as np
import src.utils as u
from src.covariance_estimator import CovarianceEstimator
from src.harmonic_info import HarmonicInfo
from src.f0_manager import F0ChangeAmount
from src.cyclic_mwf import CyclicMWF
from src.beamformer import Beamformer

eps = 1e-15


class BeamformerManager:

    def __init__(self, beamformers_names, sig_shape_k_m, minimize_noisy_cov_mvdr=True, loadings=None, noise_var_rtf=0):

        self.beamformers_names = beamformers_names
        self._harmonic_info: Optional[HarmonicInfo] = None

        if loadings is None:
            # (min_val, max_val, max_condition_number)
            loadings = {'mvdr': (0, 0, 1000), 'mwf': (0, 0, 1000)}

        self.mwf, self.mvdr, self.other_bf = None, None, None
        if any('mwf' in bf for bf in beamformers_names):
            self.mwf = CyclicMWF(loadings['mwf'], sig_shape_k_m)

        if any('mf' in bf for bf in beamformers_names):
            self.other_bf = Beamformer(loadings, sig_shape_k_m)

    @property
    def harmonic_info(self):
        return self._harmonic_info

    @harmonic_info.setter
    def harmonic_info(self, value: HarmonicInfo):
        """ Update the instantiated beamformers, too """

        self._harmonic_info = value
        if self.mwf:
            self.mwf.harmonic_info = value
        if self.mvdr:
            self.mvdr.harmonic_info = value

    def compute_weights_all_beamformers(self, cov_dict, rtf_oracle=np.array([]), idx_chunk=0, name_input_sig='noisy'):
        # which_bins_cyclic_bfs: indices of the bins for which cyclic beamformers are computed

        assert self.harmonic_info, "harmonic_info must be set before beamforming"

        error_flags = {}
        weights = {}
        speech_rtf_oracle = copy.deepcopy(rtf_oracle)
        which_bins_cyclic_bfs = self.harmonic_info.harmonic_bins

        cond_num_cov = {}
        singular_values = {}

        for bf_name in self.beamformers_names:

            bf_first_name, bf_variant = bf_name.split('_')
            if bf_first_name == 'mvdr':

                self.mvdr.rtf_needs_estimation = self.mvdr.check_if_rtf_needs_estimation(idx_chunk)
                weights[bf_name], error_flags[bf_name], cond_num_cov[bf_name], singular_values[bf_name] = (
                    self.mvdr.compute_mvdr_beamformers(cov_dict[name_input_sig + '_nb'], cov_dict['noise_nb'], bf_variant, speech_rtf_oracle))

            elif bf_first_name == 'cmvdr':

                self.mvdr.rtf_needs_estimation = self.mvdr.check_if_rtf_needs_estimation(idx_chunk)
                weights[bf_name], error_flags[bf_name], cond_num_cov[bf_name], singular_values[bf_name] = (
                    self.mvdr.compute_cyclic_mvdr_beamformers(cov_dict, bf_variant, which_bins_cyclic_bfs, speech_rtf_oracle,
                                                              name_input_sig=name_input_sig))

            elif bf_first_name == 'mwf':
                weights[bf_name], error_flags[bf_name] = (
                    self.mwf.compute_narrowband_mwf_beamformers(cov_dict, bf_variant))

            elif bf_first_name == 'cmwf':
                # print("DEBUG keep P eigenvalues for the noise covariance matrix (cMWF)")
                weights[bf_name], error_flags[bf_name] = (
                    self.mwf.compute_cyclic_mwf_beamformers(cov_dict, bf_variant, processed_bins=which_bins_cyclic_bfs))

            elif bf_first_name == 'mf' and bf_variant == 'semi-oracle':  # matched filter
                weights[bf_name] = self.other_bf.compute_matched_filter(cov_dict, speech_rtf_oracle)
            else:
                warnings.warn(f"Unknown beamformer: {bf_name}")

        # self.debug_plots_weights_cond_num_cov(weights, cond_num_cov, singular_values)

        return weights, error_flags

    def use_old_weights_if_error(self, weights_, weights_old, error_flags):
        # Discard the new weights if there is an error in the computation
        # TODO change to support cmvdr
        raise NotImplementedError

        if not weights_ or not weights_old:
            return weights_

        M, K_nfft = weights_[list(weights_.keys())[0]].shape
        weights = copy.deepcopy(weights_)
        for key in weights.keys():
            bf_first_name, bf_variant = key.split('_')
            alternative_bf = '_'.join(['mwf', bf_variant])
            key_from = alternative_bf if 'cmwf' in key else key
            if weights_old[key_from].shape[0] != M:
                print(
                    f"Old weights for {key_from} have different shape than the new ones: {M} vs {weights_old[key_from].shape[0]}")
                continue

            if any(error_flags[key]):
                for kk in range(K_nfft):
                    if error_flags[key][kk]:
                        # keep the old weights but not for modulations (f0 might change)
                        # weights[key][M:, kk] = 0
                        weights[key][:M, kk] = weights_old[key_from][:M, kk]

        return weights

    @staticmethod
    def make_weights_dict_symmetric_around_central_frequency(K_nfft, weights_dict):
        # Expected shape of weights: (M, K_nfft)
        for key in weights_dict.keys():
            weights_dict[key][:, K_nfft // 2 + 1:] = weights_dict[key][:, 1:K_nfft // 2][:, ::-1].conj()
            weights_dict[key][:, K_nfft // 2] = np.real(weights_dict[key][:, K_nfft // 2])

            # if not u.corresponds_to_real_time_domain_signal(weights[key][..., np.newaxis]):
            #     raise ValueError(f"Beamformer {key} does not correspond to a real time-domain signal.")

        return weights_dict

    @staticmethod
    def make_signals_dict_symmetric_around_central_frequency(K_nfft, signals_dict):
        # Expected shape of signals: (K_nfft, num_frames)
        for key in signals_dict.keys():
            signals_dict[key][K_nfft // 2 + 1:] = signals_dict[key][1:K_nfft // 2][::-1].conj()
            signals_dict[key][K_nfft // 2] = np.real(signals_dict[key][K_nfft // 2])

            if not u.corresponds_to_real_time_domain_signal(signals_dict[key][np.newaxis, ...]):
                raise ValueError(f"Signal {key} does not correspond to a real time-domain signal.")

        return signals_dict

    def beamform_signals(self, noisy_stft, noisy_mod_stft_3d, slice_frames, weights,
                         mod_amount=F0ChangeAmount.no_change):

        for bf_name, weight in weights.items():
            if np.allclose(weight, 0) and slice_frames.stop - slice_frames.start > 1:
                print(f"Light warning: beamformer {bf_name} is 0 (for a single chunk).")

        # if we are towards the end of the signal, the actual number of frames might be less than the expected
        # noisy_stft is always provided in full
        L3_num_frames = min(slice_frames.stop - slice_frames.start, noisy_stft.shape[-1] - slice_frames.start)
        K_nfft_real = noisy_stft.shape[1]

        if K_nfft_real % 2 == 0:
            warnings.warn(f"{K_nfft_real = }, but it should be given by K // 2 + 1 which is an odd number. Odd!")

        # Apply cyclic beamformers to modulated signals, stationary beamformers to the normal STFT signal
        bfd = {}
        for bf_name in self.beamformers_names:

            if bf_name not in weights:
                continue

            is_narrowband = self.is_narrowband_beamformer(bf_name)
            bfd_stft = np.zeros((K_nfft_real, L3_num_frames), dtype=np.complex128, order='F')
            M = noisy_stft.shape[0]

            # if is_narrowband or mod_amount != F0ChangeAmount.no_change:
            if is_narrowband:
                for kk in range(K_nfft_real):
                    bfd_stft[kk] = weights[bf_name][:M, kk].conj().T @ noisy_stft[:, kk, slice_frames]
            else:
                # Cyclic beamformers process the modulated signals (noisy_mod_stft_3d).
                # Check how many modulations have been used to compute the weights per each frequency bin.
                # Based on that, we select the corresponding weights and apply them to the modulated signals.
                for kk in range(K_nfft_real):
                    harmonic_set_idx, P = self.harmonic_info.get_harmonic_set_and_num_shifts(kk)
                    if np.any(self.harmonic_info.harmonic_sets_before_coh):  # local coherence filtering only
                        harmonic_set_idx, _ = self.harmonic_info.get_harmonic_set_and_num_shifts(kk, before_coherence=True)

                    # If fundamental frequency is changing, ignore harmonic components when beamforming
                    if mod_amount == F0ChangeAmount.large:
                        harmonic_set_idx = 0  # could be any number! P=1 always selects the non-modulated data
                        P = 1

                    sel_signal = harmonic_set_idx, slice(M * P), kk, slice_frames
                    sel_weights = slice(M * P), kk
                    processed_sig = noisy_mod_stft_3d[sel_signal]
                    if 'wl' in bf_name and P > 1:
                        sel_weights = slice(M * P * 2), kk
                        processed_sig = np.concatenate((noisy_mod_stft_3d[sel_signal],
                                                        np.conj(noisy_mod_stft_3d)[sel_signal]))

                    bfd_stft[kk] = np.conj(weights[bf_name][sel_weights]).T @ processed_sig

            bfd[bf_name] = bfd_stft

        return bfd

    @classmethod
    def select_correlated_columns(cls, cov_noisy_wb_kk, cov_noise_wb_kk, min_correlation, M):
        if min_correlation <= 0:
            return cov_noisy_wb_kk, cov_noise_wb_kk, np.arange(cov_noisy_wb_kk.shape[0])

        cov_p = CovarianceEstimator.get_spectral_cov_from_spectral_spatial_cov(cov_noisy_wb_kk, M)
        corr_first_row = CovarianceEstimator.estimate_correlation(cov_p)

        # From the first row of the correlation matrix, keep columns corresponding to values > min_correlation
        correlated_cols_p = np.where(corr_first_row > min_correlation)[0]
        correlated_cols_p = np.unique(np.concatenate([[0], correlated_cols_p]))

        # Change indices to reflect the fact that cov_wet_est has size PM x PM, not only P x P
        # so if correlated columns is [0,1,3], we need to keep [0,1,..,M, M+1,..,2M, 3M, 3M+1,..,3M+M]
        correlated_cols_pm = np.concatenate([np.arange(M) + M * cc for cc in correlated_cols_p])

        cov_noisy_wb_kk_high_corr = copy.deepcopy(cov_noisy_wb_kk[correlated_cols_pm][:, correlated_cols_pm])
        cov_noise_wb_kk_high_corr = copy.deepcopy(cov_noise_wb_kk[correlated_cols_pm][:, correlated_cols_pm])

        return cov_noisy_wb_kk_high_corr, cov_noise_wb_kk_high_corr, correlated_cols_pm

    @staticmethod
    def check_beamformed_signals_non_zero(bfd_all_chunks_stft, signals_unproc):
        # Check if the beamformed signals are all zeros
        for bfd_name, bfd_single in bfd_all_chunks_stft.items():
            if np.allclose(bfd_single, 0) and np.any(signals_unproc['wet_rank1']['stft'][0].real > 1e-6):
                print(f"Beamformed signal {bfd_name} is all zeros.")

    @staticmethod
    def get_beamformers_names(beamformers_names_orig, selected_variants):
        # Add the beamformers to the dictionary, to have mvdr_oracle, mvdr_semi-oracle, mvdr_blind, etc.
        bf_names = {}
        for key in beamformers_names_orig.keys():
            for variant in selected_variants:
                bf_names[f"{key}_{variant}"] = beamformers_names_orig[key]

        # Sort the beamformers such that first all the 'blind' beamformers are processed, then 'semi-oracle', and finally 'oracle'
        # Also, first should be MVDR, then MWF, and finally CMWF
        # So the final order should be: 'mvdr_blind', 'mwf_blind', 'cmwf_blind', 'mvdr_semi-oracle', 'mwf_semi-oracle', 'cmwf_semi-oracle',
        # 'mvdr_oracle', 'mwf_oracle', 'cmwf_oracle'
        bf_names_keys = sorted(bf_names.keys(), key=lambda x: (
            selected_variants.index(x.split('_')[1]), list(beamformers_names_orig).index(x.split('_')[0])))
        bf_names = {key: bf_names[key] for key in bf_names_keys}

        return bf_names

    @staticmethod
    def infer_beamforming_methods(bf_config):
        """
        Infer the beamforming methods from the configuration file. Based on the selected beamformers and variants.
        For example, if the user selects 'mvdr' and 'mwf' beamformers, and 'blind', 'semi-oracle' variants,
        the methods will be 'mvdr_blind', 'mwf_blind', 'mvdr_semi-oracle', 'mwf_semi-oracle'.
        """

        bf_names = BeamformerManager.get_beamformers_names(bf_config['methods_dict'], bf_config['variants'])
        bf_names_selected = [bf_name for bf_name in bf_names.keys() if bf_names[bf_name]]

        return bf_names_selected

    def debug_plots_weights_cond_num_cov(self, weights, cond_num_cov, singular_values):

        # Condition number of the covariance matrix
        titles = list(cond_num_cov.keys())
        values = list(cond_num_cov.values())
        u.plot(values, titles=titles, transform=None, title='Condition number of the covariance matrix',
               subplot_height=1.5, time_axis=False)

        # Singular values of the covariance matrix
        kk = self.harmonic_info.harmonic_bins[0]
        singular_values_kk = list(singular_values.values())
        max_val = np.max([np.max(sv[kk]) for sv in singular_values_kk])
        singular_values_kk = [sv[kk] / max_val for sv in singular_values_kk]
        u.plot(singular_values_kk, titles=titles, transform=None, title='Singular values of the covariance matrix',
               subplot_height=1.5, time_axis=False, plot_config={'marker': 'o'})

        # Weights
        u.plot_matrix(weights['mvdr_semi-oracle'][:, :30], title='MVDR oracle weights', log=False)
        u.plot_matrix(weights['mvdr_blind'][:, :30], title='MVDR weights', log=False)
        u.plot_matrix(weights['cmvdr_semi-oracle'][:, :30], title='cMVDR oracle weights', log=False)
        u.plot_matrix(weights['cmvdr_blind'][:, :30], title='cMVDR weights', log=False)

    @staticmethod
    def is_narrowband_beamformer(bf_name):
        cyclic_beamformers = ['cmvdr', 'clcmv', 'cmwf', 'cmvdr-wl']
        return bf_name.split('_')[0] not in cyclic_beamformers
