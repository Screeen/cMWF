import numpy as np
import scipy
import src.utils as u
from src.beamformer import Beamformer

eps = 1e-15


class CyclicMWF(Beamformer):

    def __init__(self, loadings_cfg, sig_shape_k_m):
        super().__init__(loadings_cfg, sig_shape_k_m)
        self.mu = 1  # For parametric multichannel Wiener filter. Higher values give more emphasis to the noise reduction
        self.harmonic_info = None

    def compute_narrowband_mwf_beamformers(self, cov_dict, which_variant='blind'):
        # MWF beamformer (narrowband). Can be oracle, semi-oracle, or blind.

        K_nfft, M = cov_dict['noisy_nb'].shape[:2]
        K_nfft_real = K_nfft if K_nfft % 2 != 0 else K_nfft // 2 + 1
        weights_mwf = np.zeros((M, K_nfft), dtype=np.complex128)
        error_flag = np.zeros(K_nfft, dtype=bool)

        loadings = self.get_loading_nb(cov_dict, *self.loadings_cfg)

        if which_variant == 'oracle':
            for kk in range(K_nfft_real):
                weights_mwf[:, kk] = scipy.linalg.solve(cov_dict['noisy_nb'][kk] + loadings[kk] * np.eye(M),
                                                        cov_dict['cross_noisy_early_nb'][kk],
                                                        assume_a='pos')
        elif which_variant == 'semi-oracle':
            for kk in range(K_nfft_real):
                weights_mwf[:, kk] = scipy.linalg.solve(
                    cov_dict['wet_nb'][kk] + self.mu * cov_dict['noise_nb'][kk] + loadings[kk] * np.eye(M), cov_dict['wet_nb'][kk][:, 0],
                    assume_a='pos')
        elif which_variant == 'blind':

            # Blind multichannel Wiener filter (MWF) beamformer (narrowband, ALL frequencies)
            K_nfft, M = cov_dict['noisy_nb'].shape[:2]
            K_nfft_real = K_nfft if K_nfft % 2 != 0 else K_nfft // 2 + 1
            weights_mwf = np.zeros((M, K_nfft), dtype=np.complex128)
            error_flag = np.zeros(K_nfft, dtype=bool)
            loadings = self.get_loading_nb(cov_dict, *self.loadings_cfg)

            for kk in range(K_nfft_real):
                weights_mwf[:, kk], error_flag[kk] = self.compute_narrowband_mwf_beamformer_blind_single_freq(
                    cov_dict['noisy_nb'][kk], cov_dict['noise_nb'][kk], mu=self.mu, loading=loadings[kk])

        else:
            raise ValueError(f"Unknown variant for MWF beamformer: {which_variant}")

        return weights_mwf, error_flag

    def compute_narrowband_mwf_beamformer_blind_single_freq(self, cov_noisy_nb_kk, cov_noise_nb_kk, mu=1, loading=0):
        # Blind multi-channel Wiener filter (MWF) beamformer (narrowband, single frequency)

        # Estimate target signal covariance matrix (rank-1 approximation)
        M = cov_noisy_nb_kk.shape[0]
        ea, ev = scipy.linalg.eigh(cov_noisy_nb_kk + loading * np.eye(M), cov_noise_nb_kk)
        ev_left = cov_noise_nb_kk @ ev
        ev_left_signal = ev_left[:, -1:]
        ea_signal = np.diag(np.maximum(eps, ea[-1:] - 1))
        cov_wet_est = ev_left_signal @ ea_signal @ np.conj(ev_left_signal).T

        weights_mwf_kk = scipy.linalg.solve(cov_wet_est + mu * cov_noise_nb_kk + loading * np.eye(M),
                                            cov_wet_est[:, 0], assume_a='pos')
        error_flag_kk = np.max(ea) < 1

        return weights_mwf_kk, error_flag_kk

    # Cyclic MWF beamformers
    def compute_cyclic_mwf_beamformers(self, cov_dict, which_variant='blind', processed_bins=np.array([])):
        # MWF beamformer (cyclic, multiband, wideband)

        K_nfft, M = cov_dict['noisy_nb'].shape[:2]
        K_nfft_real = K_nfft if K_nfft % 2 != 0 else K_nfft // 2 + 1
        P_max = cov_dict['noisy_wb'].shape[-1] // M
        error_flag = np.zeros(K_nfft, dtype=bool)
        weights_mwf_cyclic = np.zeros((M * P_max, K_nfft), dtype=np.complex128, order='F')

        P_all = self.harmonic_info.get_num_shifts_all_frequencies()
        loadings_wb = self.get_loading_wb(cov_dict, *self.loadings_cfg, P_all)

        if which_variant == 'oracle':  # oracle MWF: uses cov_noisy_wb and cross_cov_noisy_early_wb from ground truth
            for kk in range(K_nfft_real):
                MP = M * P_all[kk] if kk in processed_bins else M
                try:
                    weights_mwf_cyclic[:MP, kk] = scipy.linalg.solve(
                        cov_dict['noisy_wb'][kk, :MP, :MP] + loadings_wb[kk] * np.eye(MP),
                        cov_dict['cross_noisy_early_wb'][kk, :MP],
                        assume_a='pos')
                except np.linalg.LinAlgError:
                    # print(f"LinAlgError for bin {kk} in cMWF oracle.")
                    error_flag[kk] = True
                    weights_mwf_cyclic[:MP, kk] = scipy.linalg.solve(
                        cov_dict['noisy_wb'][kk, :MP, :MP] + loadings_wb[kk] * np.eye(MP),
                        cov_dict['cross_noisy_early_wb'][kk, :MP],
                        assume_a='her')

        elif which_variant == 'semi-oracle':  # semi-oracle MWF: uses cov_wet and cov_noise from ground truth
            for kk in range(K_nfft_real):
                # Use all MP channels if the bin is a harmonic bin, otherwise use only M channels
                MP = M * P_all[kk] if kk in processed_bins else M

                # Ignore spectral correlations for the noise
                cov_noise_wb_kk = cov_dict['noise_wb'][kk, :MP, :MP]
                cov_noise_wb_kk = u.extract_block_diag(cov_noise_wb_kk, M)
                cov_wet_wb_kk = cov_dict['wet_wb'][kk, :MP, :MP]

                try:
                    # Compute the weights by solving the linear system
                    weights_mwf_cyclic[:MP, kk] = (
                        scipy.linalg.solve(cov_wet_wb_kk + cov_noise_wb_kk + loadings_wb[kk] * np.eye(MP),
                                           cov_wet_wb_kk[:MP, 0],
                                           # ch.cross_wet_early_wb[kk, :MP],
                                           assume_a='pos'))
                except np.linalg.LinAlgError:
                    error_flag[kk] = True
                    weights_mwf_cyclic[:MP, kk] = (
                        scipy.linalg.solve(cov_wet_wb_kk + cov_noise_wb_kk + loadings_wb[kk] * np.eye(MP),
                                           cov_wet_wb_kk[:MP, 0],
                                           assume_a='her'))

        elif which_variant == 'blind':
            weights_mwf_cyclic, error_flag = self.compute_cyclic_mwf_beamformer_blind(cov_dict, processed_bins=processed_bins)

        else:
            raise ValueError(f"Unknown variant for cyclic MWF beamformer: {which_variant}")

        return weights_mwf_cyclic, error_flag

    def compute_cyclic_mwf_beamformer_blind(self, cov_dict, processed_bins=np.array([]), mu=1):
        """ cMWF beamformer """
        # Blind multi-channel Wiener filter (MWF) beamformer (cyclic, multiband)
        # min_correlation: minimum correlation between the signals to be considered as correlated
        # processed_bins: indices of the bins for which cyclic beamformers are computed
        # mu: regularization parameter

        K_nfft, M = cov_dict['noisy_nb'].shape[:2]
        K_nfft_real = K_nfft if K_nfft % 2 != 0 else K_nfft // 2 + 1
        P_max = cov_dict['noisy_wb'].shape[-1] // M
        weights_mwf_cyclic = np.zeros((M * P_max, K_nfft), dtype=np.complex128, order='F')
        error_flag = np.zeros(K_nfft, dtype=bool)

        # Loadings are different between 'normal' and 'cyclic' variants because they are computed from different
        # covariance matrices
        P_all = self.harmonic_info.get_num_shifts_all_frequencies()
        loadings_wb = self.get_loading_wb(cov_dict, *self.loadings_cfg, P_all)
        loadings_nb = self.get_loading_nb(cov_dict, *self.loadings_cfg)

        for kk in range(K_nfft_real):
            P = P_all[kk]
            MP = M * P
            if P == 1 or kk not in processed_bins:
                weights_mwf_cyclic[:M, kk], error_flag[kk] = (
                    self.compute_narrowband_mwf_beamformer_blind_single_freq(
                        cov_dict['noisy_nb'][kk], cov_dict['noise_nb'][kk], mu=mu, loading=loadings_nb[kk]))
            else:
                # Ignore spectral correlations for the noise
                cov_noise_wb_kk = cov_dict['noise_wb'][kk, :MP, :MP]
                cov_noise_wb_kk = u.extract_block_diag(cov_noise_wb_kk, M)

                # Max rank determined by P
                max_rank_cov_wet = P_all[kk]
                # max_rank_cov_wet = 1

                cov_noisy_wb_kk = cov_dict['noisy_wb'][kk, :MP, :MP]

                # Whitening procedure
                ea, ev = scipy.linalg.eigh(cov_noisy_wb_kk + loadings_wb[kk] * np.eye(MP),
                                           cov_noise_wb_kk + eps * np.eye(MP))
                error_flag[kk] = np.max(ea) < 1
                ev_left = cov_noise_wb_kk @ ev
                ev_left_signal = ev_left[:, -max_rank_cov_wet:]
                ea_signal = np.diag(np.maximum(eps, ea[-max_rank_cov_wet:] - 1))
                cov_wet_est = ev_left_signal @ ea_signal @ np.conj(ev_left_signal).T

                # weights_mwf_cyclic[correlated_indices, kk] = scipy.linalg.solve(
                #     cov_wet_est + mu * cov_noise_wb_kk + loadings[kk] * np.eye(cov_wet_est.shape[0]),
                #     cov_wet_est[:, 0], assume_a='pos')
                try:
                    weights_mwf_cyclic[:MP, kk] = scipy.linalg.solve(cov_noisy_wb_kk + loadings_wb[kk] * np.eye(MP),
                                                                     cov_wet_est[:MP, 0], assume_a='pos')
                except np.linalg.LinAlgError:
                    weights_mwf_cyclic[:MP, kk] = scipy.linalg.solve(cov_noisy_wb_kk + loadings_wb[kk] * np.eye(MP),
                                                                     cov_wet_est[:MP, 0], assume_a='her')

        return weights_mwf_cyclic, error_flag

    @staticmethod
    def get_loading_nb(cov_dict, min_val, max_val, condition_number=1000, P_all_freqs=np.array([]), M=None):
        # Loadings for reducing condition number of the covariance matrix

        # Higher target power -> higher loadings
        # Check also "Harmonic beamformers for speech enhancement and dereverberation in the time domain" eq 59
        # where they do tr(A)/N_A * gamma
        loadings = np.trace(np.maximum(0, cov_dict['noisy_nb'] - cov_dict['noise_nb']), axis1=-2, axis2=-1).real
        loadings = np.maximum(np.minimum(loadings, max_val), min_val)

        return loadings

    @staticmethod
    # loadings_wb = self.get_loading_wb(which_variant, cov_dict, *self.loadings_cfg, P_all)
    def get_loading_wb(cov_dict, min_val, max_val, condition_number=1000, P_all_freqs=np.array([]), M=None):
        # Loadings for reducing condition number of the covariance matrix

        loadings = np.trace(np.maximum(0, cov_dict['noisy_wb'] - cov_dict['noise_wb']), axis1=-2, axis2=-1).real
        loadings = np.maximum(np.minimum(loadings, max_val), min_val)

        return loadings
