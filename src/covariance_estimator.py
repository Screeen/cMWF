import warnings

import numpy as np
from . import globs as g
from .f0_manager import F0ChangeAmount
from .harmonic_info import HarmonicInfo
from . import utils as u
from .covariance_holder import CovarianceHolder


class CovarianceEstimator:
    def __init__(self, cfg_cov_est, cyclostationary_target: bool = False,
                 harmonic_info: HarmonicInfo = None, subtract_mean: bool = False,
                 use_pseudo_cov: bool = False):

        self.ch_cache = None
        self.sig_shape_k_m_p = (-1, -1, -1)  # number of frequencies, number of microphones, number of modulations

        self.use_rank1_model_for_oracle_cov_wet_es = cfg_cov_est['use_rank1_model_for_oracle_cov_wet_estimation']
        self.recursive_average = cfg_cov_est['recursive_average']
        self.forgetting_factor = cfg_cov_est['cov_est_forgetting_factor']
        self.cyclostationary_target = cyclostationary_target

        self.harmonic_info = harmonic_info
        self.subtract_mean = subtract_mean

        self.use_pseudo_cov = use_pseudo_cov  # pseudo covariance is needed for widely linear beamforming

        if self.subtract_mean and self.cyclostationary_target:
            warnings.warn("Mean subtraction only implemented for NOISY cov for cMVDR!")

        if self.use_pseudo_cov and self.cyclostationary_target:
            raise NotImplementedError(
                "Pseudo-covariance (widely linear estimator) only implemented for NOISY cov for cMVDR!")

    def estimate_covariances(self, slice_frames, signals_dict, cov_dict_prev: dict,
                             num_mics_changed: bool, modulation_amount: str, name_input_sig='noisy'):
        """ Estimate the covariance matrices for the current chunk of signal. """

        # is_first_iteration = ch_previous.is_empty()
        is_first_iteration = cov_dict_prev == {}
        if self.recursive_average:
            # TODO: Only estimate for spectral bins in harmonic_bins
            if self.use_pseudo_cov:
                raise NotImplementedError("Recursive + widely linear not implemented")
            # Prepare covariances:
            # At first iteration, or if M changed, allocate space for the covariance matrices.
            # Then estimate noise cov. from a separate noise-only signal. Copy noise matrix into noisy matrix.
            # The rest of the matrices are initialized to multiples of the identity matrix.
            if is_first_iteration or num_mics_changed:
                cov_dict_prev = self.prepare_covariances(signals_dict, needs_initialization=True,
                                                         name_input_sig=name_input_sig)

            # Update the covariance matrices with the new signals
            cov_dict = self.rank1_update_covariances(cov_dict_prev, signals_dict, slice_frames,
                                                      forget_factor=self.forgetting_factor,
                                                      is_cmvdr=not self.cyclostationary_target)
        else:  # batch processing
            # cov_dict = ch_previous.get_as_dict()
            cov_dict = cov_dict_prev
            if is_first_iteration or modulation_amount != F0ChangeAmount.no_change or num_mics_changed:
                cov_dict = self.prepare_covariances(signals_dict, needs_initialization=False,
                                                     name_input_sig=name_input_sig)
            cov_dict = self.estimate_covariances_block_processing(cov_dict, signals_dict, slice_frames,
                                                                   name_input_sig)

        cov_dict = self.copy_multiband_to_narrowband(cov_dict, M=signals_dict[name_input_sig]['stft'].shape[0],
                                                      name_input_sig=name_input_sig)

        if modulation_amount != F0ChangeAmount.no_change or num_mics_changed:
            CovarianceEstimator.check_valid_covariances(cov_dict)

        # return CovarianceHolder(**cov_dict)
        return cov_dict

    @staticmethod
    def is_num_modulations_increased(ch_previous: CovarianceHolder, num_modulations: int):
        return ch_previous.noisy_wb.shape[-1] // ch_previous.noisy_nb.shape[-1] <= num_modulations

    @staticmethod
    def get_spectral_cov_from_spectral_spatial_cov(cov_spectral_spatial, num_mics):
        """
        Extract the spectral covariance matrix from the spectral-spatial covariance matrix.
        For example, for num_mics = 2 and num_freqs = 3, we would have to extract the X and discard the O, as in:

        X 0 | X 0 | X 0
        0 0 | 0 0 | 0 0
        ---------------------
        X 0 | X 0 | X 0
        0 0 | 0 0 | 0 0
        ---------------------
        X 0 | X 0 | X 0
        0 0 | 0 0 | 0 0

        Create a mask that has True where the X are and False where the O are. Then use the mask to extract the X.
        """
        if cov_spectral_spatial is None:
            return None

        return cov_spectral_spatial[::num_mics, ::num_mics]

    @staticmethod
    def estimate_correlation(cov):

        num_freqs = cov.shape[0]

        # compute correlation coefficients (only for the upper triangular part)
        corr_coeffs = np.ones((num_freqs, num_freqs)) * np.nan
        for k1 in range(num_freqs):
            for k2 in range(k1 + 1, num_freqs):
                corr_coeffs[k1, k2] = np.abs(np.real(cov[k1, k2] / np.sqrt(cov[k1, k1] * cov[k2, k2])))

        if np.all(np.isnan(corr_coeffs)):
            raise ValueError(
                "All correlation coefficients are nan. This is likely due to a covariance matrix with zeros.")

        # Make corr_coeffs symmetric by replacing the nans in the lower triangular part with the corresponding values
        corr_coeffs = np.where(np.isnan(corr_coeffs), corr_coeffs.T, corr_coeffs)

        # Replace the nans in the diagonal with 1
        corr_coeffs[np.diag_indices(num_freqs)] = 1

        return corr_coeffs[0, :]

    @staticmethod
    def add_error_to_covariances(ch: CovarianceHolder, error: float):
        # add white noise with variance error to the covariance matrices above
        # var = (1/12) * (high - low)^2 = (1/12) * high^2
        # high = sqrt( 12 * var ) = 2 * sqrt(3) * var

        max_err = 2 * np.sqrt(3) * error
        # ch.var_early = ch.var_early + g.rng.uniform(high=max_err, size=ch.var_early.shape)

        # var_wet_nb are the elements of num_freqs diagonal matrices of size num_mics x num_mics
        num_freqs, num_mics, _ = ch.wet_nb.shape
        var_wet_nb = g.rng.uniform(high=max_err, size=num_freqs * num_mics)
        for kk in range(num_freqs):
            ch.wet_nb[kk] = ch.wet_nb[kk] + np.diag(var_wet_nb[kk * num_mics:(kk + 1) * num_mics])

        # var_wet_wb are the elements of num_freqs diagonal matrices of size num_mics_p x num_mics_p
        _, num_mics_p, _ = ch.wet_wb.shape
        var_wet_wb = g.rng.uniform(high=max_err, size=num_freqs * num_mics_p)
        for kk in range(num_freqs):
            ch.wet_wb[kk] = ch.wet_wb[kk] + np.diag(var_wet_wb[kk * num_mics_p:(kk + 1) * num_mics_p])

        ch.cross_cov_noisy_early_wb = ch.cross_noisy_early_wb + g.rng.uniform(high=max_err,
                                                                              size=ch.cross_noisy_early_wb.shape)
        # ch.cross_cov_wet_early_wb = ch.cross_wet_early_wb + utils.rng.uniform(high=max_err,
        #                                                                       size=ch.cross_wet_early_wb.shape)

        ch.cross_cov_noisy_early_nb = ch.cross_noisy_early_nb + g.rng.uniform(high=max_err,
                                                                              size=ch.cross_noisy_early_nb.shape)

        return ch

    @staticmethod
    def allocate_covariance_matrices(sig_shape_k_m_p, is_mwf=True, use_pseudo_cov=False, name_input_sig='noisy'):
        # Allocate the covariance matrices

        K_nfft, M, P = sig_shape_k_m_p
        cov_dict = {
            name_input_sig + '_wb': np.zeros((K_nfft, M * P, M * P), dtype=np.complex128, order='F'),
            'noise_wb': np.zeros((K_nfft, M * P, M * P), dtype=np.complex128, order='F'),
        }

        if is_mwf:
            cov_dict.update({
                'wet_wb': np.zeros((K_nfft, M * P, M * P), dtype=np.complex128, order='F'),
                'cross_noisy_early_wb': np.zeros((K_nfft, M * P), dtype=np.complex128, order='F'),
            })

        if use_pseudo_cov:
            cov_dict.update({
                name_input_sig + '_pseudo': np.zeros((K_nfft, M * P, M * P), dtype=np.complex128, order='F'),
            })

        return cov_dict

    @staticmethod
    def initialize_covariance_matrices(cov_dict, scaling=None, name_input_sig='noisy'):

        input_wb = name_input_sig + '_wb'
        MP = cov_dict[input_wb].shape[-1]

        initialize_with_identity = []
        initialize_with_selection = []
        copy_from_noise = [input_wb]
        skip_initialization = ['noise_wb', 'wet_wb', 'cross_noisy_early_wb', 'cross_wet_early_wb',
                               name_input_sig + '_pseudo']  # leave as all zeros
        nfft = cov_dict[input_wb].shape[0]

        identity_matrix = np.identity(MP, dtype=np.complex128) * scaling
        selection = np.zeros(MP, dtype=np.complex128)
        selection[g.mic0_idx] = scaling

        for key in cov_dict.keys():
            if key in skip_initialization:
                continue
            elif key in initialize_with_identity:
                for kk in range(nfft):
                    cov_dict[key][kk] = identity_matrix.copy()
            elif key in initialize_with_selection:
                for kk in range(nfft):
                    cov_dict[key][kk] = selection.copy()
            elif key in copy_from_noise:
                cov_dict[key] = cov_dict['noise_wb'].copy()
            else:
                raise ValueError(f"Unknown key {key}")

        return cov_dict

    def rank1_update_covariances(self, cov_dict, signals: dict, slice_frames,
                                 forget_factor: float, is_cmvdr: bool = False):
        """ Update the covariance matrices with the new signals (rank-1 update). """
        # Higher forgetting factor: we forget more, therefore more weight to the NEW covariance matrices.
        # Increase forgetting factor to adapt more quickly to changes in the environment.

        M, K_nfft = signals['noisy']['stft'].shape[:2]
        P = cov_dict['noisy_wb'].shape[-1] // M

        if M * P != cov_dict['noisy_wb'].shape[-1]:
            raise ValueError(
                f"Number of microphones {M} and modulations {P} changed. why was the matrix not reallocated?."
                f"Old shape: {cov_dict['noisy_wb'].shape}, new shape: {K_nfft, M * P, M * P}")

        # Update received signal covariance matrix
        for kk in range(K_nfft):
            harmonic_set_idx, P = self.harmonic_info.get_harmonic_set_and_num_shifts(kk)
            sel = harmonic_set_idx, slice(M * P), kk, slice_frames
            noisy_update = signals['noisy']['mod_stft_3d'][sel] @ signals['noisy']['mod_stft_3d_conj'][sel].T

            sel_cov = kk, slice(M * P), slice(M * P)  # corresponds to [kk, :M*P, :M*P]
            cov_dict['noisy_wb'][sel_cov] = (1. - forget_factor) * cov_dict['noisy_wb'][
                sel_cov] + forget_factor * noisy_update

        # Return early if possible: The quantities that follow are used in the cMWF beamformer, but not in the cMVDR.
        if is_cmvdr:
            return cov_dict

        for kk in range(K_nfft):
            harmonic_set_idx, P = self.harmonic_info.get_harmonic_set_and_num_shifts(kk)
            sel = harmonic_set_idx, slice(M * P), kk, slice_frames
            sel_cov = kk, slice(M * P), slice(M * P)  # corresponds to [kk, :M*P, :M*P]

            # Update noise signal covariance matrix based on modulated noise
            # TODO: Kind of wrong because 'noise_cov_est' has different length then 'noise'. So it does not really
            # TODO: make sense to use same temporal frames. But 'noise' is oracle data and should not be used here.
            noise_update = signals['noise_cov_est']['mod_stft_3d'][sel] @ \
                           signals['noise_cov_est']['mod_stft_3d_conj'][sel].T
            cov_dict['noise_wb'][sel_cov] = ((1. - forget_factor) * cov_dict['noise_wb'][sel_cov] +
                                             forget_factor * noise_update)

            # Update the target reverberant signal covariance matrix (semi-oracle cMWF)
            wet_update = signals['wet_rank1']['mod_stft_3d'][sel] @ signals['wet_rank1']['mod_stft_3d_conj'][sel].T
            cov_dict['wet_wb'][sel_cov] = (1. - forget_factor) * cov_dict['wet_wb'][
                sel_cov] + forget_factor * wet_update

            # Update the cross-covariance matrix between received signal and early reverberant signal (oracle cMWF)
            sel_xcov = kk, slice(M * P)  # corresponds to [kk, :M*P]
            cross_noisy_early_update = (signals['noisy']['mod_stft_3d'][sel] *
                                        signals['wet_rank1']['mod_stft_3d_conj'][
                                            harmonic_set_idx, g.mic0_idx, kk, slice_frames].T)
            cov_dict['cross_noisy_early_wb'][sel_xcov] = (
                        (1. - forget_factor) * cov_dict['cross_noisy_early_wb'][sel_xcov] +
                        forget_factor * np.squeeze(cross_noisy_early_update))

            # Do not update as we don't use this matrix anymore (can be read from wet directly)
            # cross_wet_early_update = (signals['wet_rank1']['mod_stft_3d'][sel] *
            #                           signals['wet_rank1']['mod_stft_3d_conj'][g.mic0_idx, kk, slice_frames].T)
            # cov_dict['cross_wet_early_wb'][kk] = ((1. - beta) * cov_dict['cross_wet_early_wb'][kk] +
            #                                       beta * np.squeeze(cross_wet_early_update))

        return cov_dict

    def estimate_covariances_block_processing(self, cov_dict, signals_dict, slice_frames, name_input_sig='noisy'):
        """ Estimate the covariance matrices (all time frames at once)"""

        M, K_nfft = signals_dict[name_input_sig]['stft'].shape[:2]
        input_wb = name_input_sig + '_wb'

        # Extract the modulated signals
        noisy_ = signals_dict[name_input_sig]
        wet_rank1 = {}
        early_stft_conj = {}
        if 'wet_rank1' in signals_dict.keys():
            wet_rank1 = signals_dict['wet_rank1']
            early_stft_conj = wet_rank1['stft_conj'][g.mic0_idx][np.newaxis]

        L2_frames_chunk = noisy_['stft'][:, 0, slice_frames].shape[-1]

        assert L2_frames_chunk >= 1

        for kk in range(K_nfft):
            harmonic_set_idx, P = self.harmonic_info.get_harmonic_set_and_num_shifts(kk)

            # Select the correct slice for the current frequency, depending on the harmonic set
            sel = harmonic_set_idx, slice(M * P), kk, slice_frames
            sel_cov = kk, slice(M * P), slice(M * P)  # corresponds to [kk, :M*P, :M*P]

            mod_stft = 'mod_stft_3d'
            mod_stft_conj = 'mod_stft_3d_conj'

            if mod_stft in noisy_.keys():
                if self.subtract_mean:
                    mean = np.mean(noisy_[mod_stft][sel], axis=1)
                    cov_dict[input_wb][sel_cov] = (
                            ((noisy_[mod_stft][sel] - mean[:, np.newaxis]) @
                             (noisy_[mod_stft_conj][sel] - np.conj(mean[:, np.newaxis])).T) / (L2_frames_chunk - 1))
                else:
                    cov_dict[input_wb][sel_cov] = (
                            (noisy_[mod_stft][sel] @ noisy_[mod_stft_conj][sel].T) / L2_frames_chunk)

                if self.use_pseudo_cov:
                    sel_cov = kk, slice(M * P), slice(M * P)
                    if self.subtract_mean:
                        mean = np.mean(noisy_[mod_stft][sel], axis=1, keepdims=True)
                        cov_dict['noisy_pseudo'][sel_cov] = (  # Pseudo-covariance: corresponds to E{xx^T} != E{xx^H}
                                ((noisy_[mod_stft][sel] - mean) @ (noisy_[mod_stft][sel].T - mean.T)) / L2_frames_chunk)
                    else:
                        cov_dict['noisy_pseudo'][sel_cov] = (  # Pseudo-covariance: corresponds to E{xx^T} != E{xx^H}
                                (noisy_[mod_stft][sel] @ noisy_[mod_stft][sel].T) / L2_frames_chunk)

            else:
                # Just compute narrowband covariance
                sel_nb = slice(M), kk, slice_frames
                sel_cov_nb = kk, slice(M), slice(M)  # corresponds to [kk, :M, :M]
                cov_dict[input_wb][sel_cov_nb] = (
                        (noisy_['stft'][sel_nb] @ noisy_['stft_conj'][sel_nb].T) / L2_frames_chunk)

            if mod_stft in wet_rank1.keys():
                if self.use_rank1_model_for_oracle_cov_wet_es:
                    cov_dict['wet_wb'][sel_cov] = (
                            (wet_rank1[mod_stft][sel] @ wet_rank1[mod_stft_conj][sel].T) / L2_frames_chunk)
                else:
                    # Use "true" reverberant signal (with late reflections) to estimate the covariance matrix
                    raise NotImplementedError("Var early gonna be wrong")

            if mod_stft in noisy_.keys() and 'cross_noisy_early_wb' in cov_dict.keys():
                sel_mic0 = g.mic0_idx, kk, slice_frames  # corresponds to [mic0_idx, kk, slice_frames]
                sel_xcov = kk, slice(M * P)  # corresponds to [kk, :M*P]
                # r_xd. Used in MWF-WB as w = (r_y)^(-1) @ r_xd
                # per each frequency kk, this is a vector
                cov_dict['cross_noisy_early_wb'][sel_xcov] = (
                        noisy_[mod_stft][sel] @ early_stft_conj[sel_mic0] / L2_frames_chunk)

            # if 'mod_stft_3d' in wet_rank1.keys():
            #     # R_ds = E[v(d) s*]
            #     cov_dict['cross_wet_early_wb'][kk] = (
            #             wet_rank1['mod_stft_3d'][sel] @ early_stft_conj[sel_mic0] / L2_frames_chunk)

        return cov_dict

    def estimate_noise_covariance(self, noise_cov_est_sig, cov_noise_wb):
        """ Estimate the noise covariance matrix based on a separate noise-only signal. """
        # Noise covariance matrix wideband needs to be updated in the loop. That's because the modulations
        # present in the matrix change from chunk to chunk. We still use the same amount of frames as for
        # the narrowband noise covariance matrix.

        # if M * P != cov_dict['noise_wb'].shape[-1]:
        #     warnings.warn("The number of microphones and modulations has changed. This is not supported.")
        #     return cov_dict

        K_nfft, M, _ = self.sig_shape_k_m_p
        L3_frames_cov_est = noise_cov_est_sig['stft'].shape[-1]
        assert L3_frames_cov_est >= 1

        if 'mod_stft_3d' in noise_cov_est_sig.keys():
            # If modulated signal is available, compute spectral-spatial covariance
            for kk in range(K_nfft):
                harmonic_set_idx, P = self.harmonic_info.get_harmonic_set_and_num_shifts(kk)

                # Select the correct slice for the current frequency, depending on the harmonic set
                sel_data = harmonic_set_idx, slice(M * P), kk, slice(None)
                sel_cov = kk, slice(M * P), slice(M * P)  # corresponds to [kk, :M*P, :M*P]

                cov_noise_wb[sel_cov] = (noise_cov_est_sig['mod_stft_3d'][sel_data] @
                                         noise_cov_est_sig['mod_stft_3d_conj'][sel_data].T) / L3_frames_cov_est
        else:
            # Wideband noise covariance matrix might not be needed, for example for cMVDR when minimizing NOISY.
            # Compute spatial covariance only:
            for kk in range(K_nfft):
                cov_noise_wb[kk, :M, :M] = (noise_cov_est_sig['stft'][:, kk] @
                                            noise_cov_est_sig['stft_conj'][:, kk].T) / L3_frames_cov_est

        return cov_noise_wb

    @staticmethod
    def check_valid_covariances(cov_dict):
        """ Check for nans and zeros in the covariance matrices """

        if not u.is_debug_mode():
            return

        for cov_name, cov in cov_dict.items():
            if np.any(np.isnan(cov)):
                print(f"Covariance matrix {cov_name} contains nans")
            if 'wet' in cov_name or 'early' in cov_name:  # wet and early can be all zeros
                continue
            if cov.size == 0:
                print(f"Covariance matrix {cov_name} is empty")
                continue

            for kk in range(cov.shape[0]):
                if np.allclose(cov[kk], 0):
                    print(f"Covariance matrix {cov_name} is all zeros at {kk = }. {np.max(np.abs(cov[kk])) = }")

    @staticmethod
    def copy_multiband_to_narrowband(cov_dict, M, name_input_sig='noisy'):
        # Infer narrowband (spatial) covariance matrices from wideband (cylic) covariance matrices

        def is_present(cov_name):
            return cov_name in cov_dict.keys() and cov_dict[cov_name].size > 0

        name_input_wb = name_input_sig + '_wb'
        name_input_nb = name_input_sig + '_nb'
        if is_present(name_input_wb):
            cov_dict[name_input_nb] = cov_dict[name_input_wb][:, :M, :M]

        if is_present('noise_wb'):
            cov_dict['noise_nb'] = cov_dict['noise_wb'][:, :M, :M]

        if is_present('wet_wb'):
            cov_dict['wet_nb'] = cov_dict['wet_wb'][:, :M, :M]

        if is_present('cross_noisy_early_wb'):
            cov_dict['cross_noisy_early_nb'] = cov_dict['cross_noisy_early_wb'][:, :M]

        if is_present('cross_wet_early_wb'):
            cov_dict['cross_wet_early_nb'] = cov_dict['cross_wet_early_wb'][:, :M]

        return cov_dict

    def set_dimensions(self, K_nfft_real_M_P):
        self.sig_shape_k_m_p = K_nfft_real_M_P

    def prepare_covariances(self, signals_dict, needs_initialization=True, name_input_sig='noisy'):
        """ Allocate, calibrate and initialize the covariance matrices. """

        cov_dict = self.allocate_covariance_matrices(self.sig_shape_k_m_p, is_mwf=self.cyclostationary_target,
                                                     use_pseudo_cov=self.use_pseudo_cov, name_input_sig=name_input_sig)
        cov_dict['noise_wb'] = self.estimate_noise_covariance(signals_dict['noise_cov_est'], cov_dict['noise_wb'])
        if needs_initialization:
            cov_dict = self.initialize_covariance_matrices(cov_dict, scaling=np.max(np.abs(cov_dict['noise_wb'])))

        # return CovarianceHolder(**cov_dict), cov_dict
        return cov_dict

    #     # Estimate the covariance matrices
    #     for kk in range(K_nfft):
    #         sel = slice(None), kk, slice_frames  # corresponds to [:, kk, slice_frames]
    #         sel_mic0 = mic0_idx, kk, slice_frames  # corresponds to [mic0_idx, kk, slice_frames]
    #
    #         # Noise covariance matrix wideband needs to be updated in the loop. That's because the modulations
    #         # present in the matrix change from chunk to chunk. We still use the same amount of frames as for
    #         # the narrowband noise covariance matrix.
    #         cov_noise_wb[kk] = (noise_cov_est_mod_stft_3d[:, kk] @
    #                             noise_cov_est_mod_stft_3d_conj[:, kk].T) / L3_frames_cov_est
    #         cov_noise_wb[kk] = cov_noise_wb[kk] + loading * np.eye(M * P)
    #
    #         ##### cov_noise_nb[kk] = copy.deepcopy(cov_noise_wb[kk][:M, :M])
    #
    #         # var_early[kk] = np.real(early_stft[sel] @ early_stft_conj[sel].T / L2_frames_chunk).item()
    #         # var_early[kk] = np.real(wet_rank1_stft[sel_mic0] @ wet_rank1_stft[sel_mic0].T / L2_frames_chunk).item()
    #
    #         if self.use_rank1_model_for_oracle_cov_wet_estimation:
    #             # cov_wet_nb[kk] = (wet_rank1_stft[sel] @ wet_rank1_stft_conj[sel].T) / L2_frames_chunk
    #
    #             # wet_rank1_mod_stft is obtained as C_rtf @ early_mod_stft
    #             # Thus, cov_wet_wb[kk] = C_rtf[kk] @ cov_early_wb[kk] @ C_rtf[kk].conj().T
    #             cov_wet_wb[kk] = (wet_rank1_mod_stft_3d[sel] @ wet_rank1_mod_stft_3d_conj[sel].T) / L2_frames_chunk
    #             # print(f"{np.allclose(wet_rank1_stft[sel], C_rtf[kk] @ early_stft[sel]) = }")
    #         else:
    #             # Use "true" reverberant signal (with late reflections) to estimate the covariance matrix
    #             raise NotImplementedError("Var early gonna be wrong")
    #             # cov_wet_nb[kk] = (wet_stft[sel] @ wet_stft_conj[sel].T) / L2_frames_chunk
    #             cov_wet_wb[kk] = (wet_mod_stft_3d[sel] @ wet_mod_stft_3d_conj[sel].T) / L2_frames_chunk
    #
    #         ##### cov_wet_nb[kk] = copy.deepcopy(cov_wet_wb[kk][:M, :M])
    #         ##### var_early[kk] = cov_wet_nb[kk][0, 0].real
    #
    #         cov_noisy_wb[kk] = (noisy_mod_stft_3d[sel] @ noisy_mod_stft_3d_conj[sel].T) / L2_frames_chunk
    #         cov_noisy_wb[kk] = cov_noisy_wb[kk] + loading * np.eye(M * P)  # for inversion stability
    #         ##### cov_noisy_nb[kk] = copy.deepcopy(cov_noisy_wb[kk][:M, :M])
    #
    #         # r_xd. Used in MWF-WB as w = (r_y)^(-1) @ r_xd
    #         cross_cov_noisy_early_wb[kk] = (noisy_mod_stft_3d[sel] @ early_stft_conj[sel_mic0] / L2_frames_chunk)
    #         # cross_cov_noisy_early_wb[kk] = (noisy_mod_stft_3d[sel] @ wet_rank1_mod_stft_3d_conj[sel_mic0] / L2_frames_chunk)
    #
    #         # for tt in range(L2_frames_chunk):
    #         #     cross_cov_noisy_early_wb[kk] += noisy_mod_stft_3d[:, kk, tt] * early_stft_conj[0, kk, tt]
    #         # cross_cov_noisy_early_wb[kk] /= L2_frames_chunk
    #
    #         # raise ValueError("This is wrong! We are not using the modulated signals for the early signal."
    #         #                  "Also check signal below! Suggestion: let's rewrite this function.")
    #         # cross_cov_noisy_early_wb[kk] = (noisy_mod_stft_3d[sel] @ early_mod_stft_3d_conj[:, kk, slice_frames] / L2_frames_chunk)
    #
    #         # speech_rtf[:, kk] * var_early[kk]
    #         # cross_cov_noisy_early_nb[kk] = (noisy_stft[sel] @ early_stft_conj[sel_mic0] / L2_frames_chunk)
    #         ##### cross_cov_noisy_early_nb[kk] = copy.deepcopy(cross_cov_noisy_early_wb[kk][:M])
    #
    #         # cross_cov_wet_early_wb[kk] = (wet_rank1_mod_stft_3d[sel] @ early_stft_conj[sel_mic0] / L2_frames_chunk)
    #         # cross_cov_wet_early_wb[kk] = (wet_rank1_mod_stft_3d[sel] @ early_mod_stft_3d_conj[sel_mic0] / L2_frames_chunk)
    #
    #         # r_dd. Used in MWF-WB as w = (r_y)^(-1) @ r_dd
    #         # cov_early_wb[kk] = (early_mod_stft_3d[sel] @ early_mod_stft_3d_conj[sel].T / L2_frames_chunk)
    #         # cross_cov_wet_early_wb[kk] = C_rtf[kk] @ cov_early_wb[kk][:, mod0_idx]
    #         # cross_cov_wet_early_wb[kk] = wet_mod_stft_3d[sel] @ early_mod_stft_3d_conj[mod0_idx, kk, slice_frames] / L2_frames_chunk
    #         # cross_cov_wet_early_wb[kk] = wet_mod_stft_3d[sel] @ wet_mod_stft_3d_conj[mod0_idx, kk, slice_frames] / L2_frames_chunk
    #         # cross_cov_wet_early_wb[kk] = (wet_mod_stft_3d[sel] @ early_stft_conj[sel_mic0] / L2_frames_chunk)
    #         # cross_cov_wet_early_wb[kk] = wet_rank1_mod_stft_3d[sel] @ early_mod_stft_3d_conj[mic0_idx, kk, slice_frames] / L2_frames_chunk
    #
    #         # Implementation 1: R_ds = E[v(d) s*]
    #         # for tt in range(L2_frames_chunk):
    #         #     # cross_cov_wet_early_wb[kk] += wet_rank1_mod_stft_3d[:, kk, tt] * early_stft_conj[0, kk, tt]
    #         #     cross_cov_wet_early_wb[kk] += wet_rank1_mod_stft_3d[:, kk, tt] * wet_rank1_mod_stft_3d_conj[0, kk, tt]
    #         # cross_cov_wet_early_wb[kk] /= L2_frames_chunk
    #
    #         # Implementation 2: R_ds = E[v(d) v(d)^H] @ np.c_[1, 0, ..., 0]
    #         # assert np.allclose(early_stft_conj[0, kk, tt], wet_rank1_mod_stft_3d_conj[mic0_idx, kk, tt])
    #         # for tt in range(L2_frames_chunk):
    #         #     temp = np.c_[wet_rank1_mod_stft_3d[:, kk, tt]] @ np.reshape(wet_rank1_mod_stft_3d_conj[:, kk, tt], (1, -1), order='F')
    #         #     cross_cov_wet_early_wb[kk] += temp[:, mic0_idx]
    #         # cross_cov_wet_early_wb[kk] /= L2_frames_chunk
    #
    #         # Implementation 3: R_ds = C @ E[v(s) s^*]
    #         # assert np.allclose(C_rtf[kk] @ early_mod_stft_3d[:, kk], wet_rank1_mod_stft_3d[:, kk])
    #         # for tt in range(L2_frames_chunk):
    #         #     cross_cov_wet_early_wb[kk] += C_rtf[kk] @ (early_mod_stft_3d[:, kk, tt] * early_stft_conj[0, kk, tt])
    #         # cross_cov_wet_early_wb[kk] /= L2_frames_chunk
    #
    #         # Implementation 4: like 1, but vectorized
    #         # cross_cov_wet_early_wb[kk] = (wet_rank1_mod_stft_3d[sel] @ wet_rank1_mod_stft_3d_conj[sel_mic0]) / L2_frames_chunk
    #
    #         # Impl 5
    #         cross_cov_wet_early_wb[kk] = (wet_rank1_mod_stft_3d[sel] @ early_stft_conj[sel_mic0] / L2_frames_chunk)
    #
    # def calibrate_and_update_noise_covariance(self, noise_cov_est_sig, ch_previous, forgetting_factor):
    #
    #     cov_dict = ch_previous.get_as_dict()
    #     cov_noise_prev = copy.deepcopy(cov_dict.get('noise_wb', np.zeros_like(cov_dict['noisy_wb'])))
    #
    #     cov_dict['noise_wb'] = self.estimate_noise_covariance(noise_cov_est_sig, cov_dict['noise_wb'])
    #
    #     if cov_noise_prev.shape == cov_dict['noise_wb'].shape and not np.allclose(cov_noise_prev, 0):
    #         cov_dict['noise_wb'] = cov_noise_prev * (1. - forgetting_factor) + cov_dict['noise_wb'] * forgetting_factor
    #
    #     return CovarianceHolder(**cov_dict)
    #
    # @staticmethod
    # def preserve_spatial_correlation(ch_old, sig_shape_k_m_p, cyclostationary_target=False):
    #     """
    #     Preserve the spatial correlation of the covariances in ch_previous by copying the spatial part of the
    #     covariance matrices to the new covariance matrices.
    #     """
    #
    #     # Check if the signal shape has changed (M or P). In that case, we need to reallocate.
    #     if not ch_old.has_compatible_size(sig_shape_k_m_p):
    #         cov_dict_new = CovarianceEstimator.allocate_covariance_matrices(sig_shape_k_m_p)
    #     else:
    #         return ch_old
    #
    #     cov_dict_old = ch_old.get_as_dict()
    #     if not cyclostationary_target:  # cMVDR
    #         # We do not need the wet and cross matrices, as they are not used in the (c)MVDR beamformer.
    #         to_copy = ['noise_wb', 'noisy_wb']
    #         to_leave_uninitialized = ['wet_wb', 'cross_noisy_early_wb', 'cross_wet_early_wb']
    #     else:  # cMWF
    #         to_copy = ['noise_wb', 'noisy_wb', 'wet_wb', 'cross_noisy_early_wb', 'cross_wet_early_wb']
    #         to_leave_uninitialized = []
    #
    #     _, M, _ = sig_shape_k_m_p
    #     for cov_name in cov_dict_new.keys():
    #         if cov_name in to_leave_uninitialized or '_nb' in cov_name:
    #             continue
    #         elif cov_name in to_copy:
    #             if cov_dict_old[cov_name].ndim == 3:
    #                 cov_dict_new[cov_name][:, :M, :M] = cov_dict_old[cov_name][:, :M, :M]
    #             elif cov_dict_old[cov_name].ndim == 2:
    #                 cov_dict_new[cov_name][:, :M] = cov_dict_old[cov_name][:, :M]
    #         else:
    #             raise ValueError(f"Unknown cov_name {cov_name}")
    #
    #         if np.allclose(cov_dict_new[cov_name], 0):
    #             warnings.warn(f"Preserved covariance matrix {cov_name} is all zeros.")
    #
    #     return CovarianceHolder(**cov_dict_new)
    #
    # @staticmethod
    # def reallocate_and_preserve_block_diagonals(ch_previous, sig_shape_k_m_p):
    #     """
    #     Preserve the spatial correlation of the covariances in ch_previous by copying the spatial part of the
    #     covariance matrices to the new covariance matrices.
    #     """
    #
    #     # Need to reallocate as M or P has changed!
    #     cov_dict_new = CovarianceEstimator.allocate_covariance_matrices(sig_shape_k_m_p)
    #     cov_dict_old = ch_previous.get_as_dict()
    #
    #     _, M, _ = sig_shape_k_m_p
    #     for cov_name in cov_dict_new.keys():
    #         if cov_dict_old[cov_name].ndim == 3:
    #             for kk in range(sig_shape_k_m_p[0]):
    #                 cov_dict_new[cov_name][kk] = u.extract_block_diag(cov_dict_old[cov_name][kk], M)
    #         elif cov_dict_old[cov_name].ndim == 2:
    #             cov_dict_new[cov_name][:, :M] = cov_dict_old[cov_name][:, :M]
    #         else:
    #             raise ValueError(f"Unknown cov_name {cov_name}")
    #
    #         if np.allclose(cov_dict_new[cov_name], 0):
    #             warnings.warn(f"Preserved covariance matrix {cov_name} is all zeros.")
    #
    #     return CovarianceHolder(**cov_dict_new)
