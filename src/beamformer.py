import numpy as np
import scipy
eps = 1e-15


class Beamformer:

    # RTF estimation
    rtf_est_warmup_chunks = 1  # number of initial chunks to estimate RTF
    rtf_est_interval_chunks = 5  # number of chunks after which RTF is re-estimated

    def __init__(self, loadings_cfg: list = None, sig_shape_k_m=None):
        self.loadings_cfg = loadings_cfg
        self.rtf_needs_estimation = np.ones(sig_shape_k_m[0], dtype=bool)

    def check_if_rtf_needs_estimation(self, idx_chunk, warmup_chunks=rtf_est_warmup_chunks,
                                      interval_chunks=rtf_est_interval_chunks):
        # By default, return all False so RTF does not get estimated
        return np.zeros_like(self.rtf_needs_estimation)

    @staticmethod
    def get_loading_nb(which_variant, cov_input_nb, min_val, max_val, condition_number=1000):
        # Loadings for reducing condition number of the covariance matrix

        loadings = np.zeros(cov_input_nb.shape[0], dtype=float)
        for kk in range(cov_input_nb.shape[0]):
            eigenvalues = np.linalg.eigvalsh(cov_input_nb[kk])
            loadings[kk] = (eigenvalues[-1] - condition_number * eigenvalues[0]) / (condition_number - 1)
        loadings = np.maximum(0, loadings)

        # Higher target power -> higher loadings
        # Check also "Harmonic beamformers for speech enhancement and dereverberation in the time domain" eq 59
        # where they do tr(A)/N_A * gamma
        # if which_variant == 'blind':
        # loadings = np.trace(np.maximum(0, ch.noisy_nb - ch.noise_nb), axis1=-2, axis2=-1).real
        # else:  # oracle or semi-oracle
        #     loadings = np.trace(ch.wet_nb, axis1=-2, axis2=-1).real

        # print("DEBUG")
        # Second option: use 10 times the noise psd as the loading
        # noise psd is the trace of the noise covariance matrix divided by the number of microphones
        # if which_variant == 'blind':
        #     loadings = np.trace(ch.noise_nb, axis1=-2, axis2=-1).real / ch.noise_nb.shape[-1]
        # else:
        #     loadings = np.ones(ch.noise_nb.shape[0]) * min_loading

        # loadings = np.minimum(max_val, np.maximum(min_val, loadings))

        return loadings

    @staticmethod
    def get_loading_wb(cov_wb, min_val, max_val, condition_number=1000, P_all_freqs=np.array([]), M=None):
        # Loadings for reducing condition number of the covariance matrix

        loadings = np.zeros(cov_wb.shape[0], dtype=float)
        for kk in range(cov_wb.shape[0]):
            P = P_all_freqs[kk] if P_all_freqs.size > 0 else 1
            MP = int(M * P)
            eigenvalues = np.linalg.eigvalsh(cov_wb[kk, :MP, :MP])
            smallest_eigenvalue = max(eigenvalues[0], 0)
            loadings[kk] = (eigenvalues[-1] - condition_number * smallest_eigenvalue) / (condition_number - 1)
        loadings = np.maximum(eps, loadings)

        return loadings

    def compute_cyclic_lcmv_beamformers(self, C_rtf_oracle, ch, which_variant, processed_bins):
        """ Compute the weights for the cyclic LCMV beamformer (cLCMV). """

        raise NotImplementedError("Update computation of modulated RTFs before using this method again.")

        K_nfft, M = ch.noisy_nb.shape[:2]
        P = ch.noisy_wb.shape[-1] // M
        weights = np.zeros((M * P, K_nfft), dtype=np.complex128, order='F')
        error_flag = np.zeros(K_nfft, dtype=bool)

        eye_nb = np.eye(M)
        eye_wb = np.eye(M * P)
        loadings = self.get_loading_nb(which_variant, ch.noisy_nb, *self.loadings_cfg['mvdr'])
        speech_rtf_oracle = C_rtf_oracle[:M, :, 0]

        for kk in range(K_nfft):
            if kk not in processed_bins:
                rtf = self.estimate_rtf_or_get_oracle_mvdr(ch.noisy_nb[kk], ch.noise_nb[kk],
                                                           speech_rtf_oracle[:, kk],
                                                           which_variant, kk)
                cov_nb_kk = ch.noisy_nb[kk] if self.minimize_noisy_cov_mvdr else ch.noise_nb[kk]
                cov_nb_kk_inv_rtf = scipy.linalg.solve(cov_nb_kk + loadings[kk] * eye_nb, rtf, assume_a='pos')
                weights[:M, kk] = np.squeeze(cov_nb_kk_inv_rtf / (np.conj(rtf).T @ cov_nb_kk_inv_rtf).real)

            else:
                C_rtf = self.estimate_shifted_rtfs_or_get_oracle_lcmv(ch.noisy_wb[kk], ch.noise_wb[kk],
                                                                      C_rtf_oracle[:, kk],
                                                                      which_variant)
                cov_wb_kk = ch.noisy_wb[kk] if self.minimize_noisy_cov_mvdr else ch.noise_wb[kk]
                const = np.zeros(P)
                const[0] = 1
                cov_wb_kk_inv_c = scipy.linalg.solve(cov_wb_kk + loadings[kk] * eye_wb, C_rtf, assume_a='pos')
                if P == 1:
                    weights[:, kk] = cov_wb_kk_inv_c / (np.conj(C_rtf).T @ cov_wb_kk_inv_c).real  # just MVDR
                else:
                    try:
                        weights[:, kk] = cov_wb_kk_inv_c @ np.linalg.inv(np.conj(C_rtf).T @ cov_wb_kk_inv_c) @ const
                    except np.linalg.LinAlgError:
                        print(f"LinAlgError for bin {kk}")
                        error_flag[kk] = True

        return weights, error_flag

    # def compute_cyclic_lcmv_beamformers_old(self, C_rtf, ch, which_variant, processed_bins):
    #
    #     K_nfft, M = ch.noisy_nb.shape[:2]
    #     P = ch.noisy_wb.shape[-1] // M
    #     weights = np.zeros((M * P, K_nfft), dtype=np.complex128, order='F')
    #     error_flag = np.zeros(K_nfft, dtype=bool)
    #
    #     eye_nb = np.eye(M)
    #     eye_wb = np.eye(M * P)
    #     loadings = self.get_loading_nb(which_variant, ch, *self.loadings['mvdr'])
    #
    #     # MVDR beamformer (narrowband)
    #     for kk in range(K_nfft):
    #         if which_variant == 'blind':
    #             continue
    #
    #         if kk not in processed_bins:
    #             rtf = self.estimate_rtf_or_get_oracle_mvdr(ch.noisy_nb[kk], ch.noise_nb[kk], C_rtf[:M, kk, 0],
    #                                                        which_variant, kk)
    #             cov_nb_kk = ch.noisy_nb[kk] if self.minimize_noisy_cov_mvdr else ch.noise_nb[kk]
    #             cov_nb_kk_inv_rtf = scipy.linalg.solve(cov_nb_kk + loadings[kk] * eye_nb, rtf, assume_a='pos')
    #             weights[:M, kk] = np.squeeze(cov_nb_kk_inv_rtf / (np.conj(rtf).T @ cov_nb_kk_inv_rtf).real)
    #
    #         else:
    #             cov_wb_kk = ch.noisy_wb[kk] if self.minimize_noisy_cov_mvdr else ch.noise_wb[kk]
    #
    #             # LCMV
    #             constraint_values = np.ones(P)
    #             cov_wb_kk_inv_c = scipy.linalg.solve(cov_wb_kk + loadings[kk] * eye_wb, C_rtf[:, kk], assume_a='pos')
    #             weights[:, kk] = cov_wb_kk_inv_c @ np.linalg.inv(
    #                 C_rtf[:, kk].conj().T @ cov_wb_kk_inv_c) @ constraint_values
    #
    #     return weights, error_flag

    def estimate_shifted_rtfs_or_get_oracle_lcmv(self, cov_noisy_wb, cov_noise_wb, C_rtf=None, which_variant='blind',
                                                 modulator=None):

        P = C_rtf.shape[-1]
        M = C_rtf.shape[0] // P
        loading_mat = np.eye(M)
        C_rtf_est = np.zeros_like(C_rtf)

        if which_variant == 'oracle' or which_variant == 'semi-oracle':  # Oracle relative transfer functions (RTFs)
            if C_rtf is None:
                raise ValueError(f"Oracle RTFs are needed if {which_variant = }")
            C_rtf_est = C_rtf

        elif which_variant == 'blind':  # Estimate RTF using GEVD (covariance whitening)
            fancy_approach = False
            for pp in range(P):
                if fancy_approach and pp > 0:
                    break
                cov_noisy_nb = cov_noisy_wb[pp * M:(pp + 1) * M, pp * M:(pp + 1) * M]
                cov_noise_nb = cov_noise_wb[pp * M:(pp + 1) * M, pp * M:(pp + 1) * M]

                ea, ev = scipy.linalg.eigh(cov_noisy_nb, cov_noise_nb + self.loadings_cfg['mvdr']['min'] * loading_mat)
                ev_left = cov_noise_nb @ ev
                ev_left_signal = ev_left[:, -1:]
                rtf_pp = ev_left_signal / ev_left_signal[0]
                C_rtf_est[pp * M:(pp + 1) * M, pp] = np.squeeze(rtf_pp)

            if fancy_approach and modulator is not None:
                raise NotImplementedError("To get time-domain we need the full RTF, not just one frequency."
                                          "This function needs to be moved up")
        else:
            raise ValueError(f"Unknown variant for MVDR beamformer: {which_variant}")

        return np.atleast_1d(np.squeeze(C_rtf_est))

    # def estimate_rtf_single_freq_cyclic(self, cov_noisy_wb, cov_noise_wb, num_mics, oracle_rtf=None,
    #                                     which_variant='blind'):
    #
    #     MP = cov_noisy_wb.shape[1]
    #     loading_mat = np.eye(MP)
    #
    #     if which_variant == 'oracle' or which_variant == 'semi-oracle':  # Oracle relative transfer functions (RTFs)
    #         if oracle_rtf is None:
    #             raise ValueError(f"Oracle RTFs are needed if {which_variant = }")
    #         rtf = oracle_rtf
    #
    #     elif which_variant == 'blind':  # Estimate RTF using GEVD (covariance whitening)
    #         ea, ev = scipy.linalg.eigh(cov_noisy_wb, cov_noise_wb + min_loading * loading_mat)
    #         ev_left = cov_noise_wb @ ev
    #         ev_left_signal = ev_left[:, -1:]
    #         rtf = ev_left_signal / ev_left_signal[0]
    #         rtf = rtf[:num_mics]
    #
    #     else:
    #         raise ValueError(f"Unknown variant for MVDR beamformer: {which_variant}")
    #
    #     return np.atleast_1d(np.squeeze(rtf))

    def compute_matched_filter(self, ch, speech_rtf_oracle):
        # Matched filter is w = Rn^-1 * rtf

        K_nfft, M = ch.noisy_nb.shape[:2]
        weights = np.zeros((M, K_nfft), dtype=np.complex128, order='F')
        for kk in range(K_nfft):
            # Non-white noise
            # weights[:, kk] = scipy.linalg.solve(ch.noise_nb[kk], speech_rtf_oracle[kk], assume_a='pos')

            # White noise
            weights[:, kk] = speech_rtf_oracle[kk]

        return weights


