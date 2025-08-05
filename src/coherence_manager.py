import copy
import warnings

import numpy as np
from scipy.signal import ShortTimeFFT

from . import globs as g
from . import utils as u
from .harmonic_info import HarmonicInfo
from .modulator import Modulator

u.set_printoptions_numpy()
u.set_plot_options()


class CoherenceManager:

    def __init__(self):
        pass

    @staticmethod
    def compute_coherence(signal, SFT: ShortTimeFFT, modulator_obj: Modulator, max_bin=-1,
                          min_relative_power=1.e+3):
        """
        Compute coherence between modulated signals and the reference signal (first microphone).

        Since the expensive step is computing the STFTs, it is no big deal if only after computing
        coherence and storing all data I remove the one with low correlation.

        - All modulations are performed
        - Coherence is calculated
        - Useless modulations are removed from signals
        - HarmonicInfo instance updated accordingly

        """

        alpha = modulator_obj.alpha_vec_hz_

        cc0 = np.where(alpha == 0)[0][0]
        if max_bin == -1:
            max_bin = int(np.ceil((3 * SFT.delta_f + np.max(np.abs(alpha))) / SFT.delta_f))

        # Modulate the data in the time domain (first microphone only)
        assert g.mic0_idx == 0
        modulated_data = modulator_obj.modulate(signal['time'][g.mic0_idx:g.mic0_idx + 1, np.newaxis, :])

        # Convert to STFT domain
        # mod_data_stft.shape = (Mics, P_sum, K_nfft, L_frames)
        # mic 0, all modulations, kk up to kk_max
        mod = SFT.stft(modulated_data)[g.mic0_idx, :, :max_bin]
        mod_c = np.conj(mod)

        # Calculate the powers at different frequency shifts (pp) for all frequency bins (kk)
        psds = np.mean(np.abs(mod) ** 2, axis=-1)  # squared magnitude averaged over time-frames
        psds = np.maximum(psds, np.max(psds) / min_relative_power)  # limit value from below to avoid artefacts when diving by this

        # rho = CoherenceManager.compute_coherence_internal(mod, mod_c, psds, alpha, cc0, SFT.delta_f, SFT.fs)
        rho = CoherenceManager.compute_coherence_internal_fast(mod, mod_c, psds, alpha, cc0, SFT.delta_f, SFT.fs)

        # by definition, but it does not always correspond to one if we limit the minimum value of the psds
        rho[cc0] = 1

        return rho

    @staticmethod
    def compute_coherence_internal(mod, mod_c, psds, alpha, cc0, delta_f, fs):
        # Calculate spectral coherence (squared)
        P_sum, kk_max, _ = mod_c.shape
        rho = np.zeros((P_sum, kk_max), dtype=float)
        for pp in range(P_sum):
            pp_hz = alpha[pp]
            for kk in range(kk_max):
                kk_hz = kk * delta_f
                if kk_hz < pp_hz or kk_hz > (fs // 2 + pp_hz):
                    # this data does not correspond to anything meaningful (unless we use complex spectrum)
                    continue
                # Coherence
                cross_psd = np.abs(np.mean(mod[cc0, kk] * mod_c[pp, kk], axis=-1)) ** 2
                # Conjugate coherence
                # cross_psd = np.abs(np.mean(mod[cc0, kk] * mod[pp, kk], axis=-1)) ** 2

                rho[pp, kk] = cross_psd / (psds[cc0, kk] * psds[pp, kk])
        return rho

    @staticmethod
    def compute_coherence_internal_fast(mod, mod_c, psds, alpha, cc0, delta_f, fs):
        # Calculate spectral coherence (squared)
        P_sum, kk_max, frames = mod_c.shape

        # 1) Build mask of valid (pp, kk) pairs
        kk_hz = np.arange(kk_max) * delta_f
        mask = (
                (kk_hz[None, :] >= alpha[:, None]) &
                (kk_hz[None, :] <= (fs / 2 + alpha[:, None]))
        )

        # 2) Compute cross-PSD for all (pp, kk)
        #    mod[cc0] has shape (kk_max, frames)
        cross = mod[cc0][None, :, :] * mod_c  # → (P_sum, kk_max, frames)
        cross_psd = np.abs(cross.mean(axis=-1)) ** 2  # → (P_sum, kk_max)

        # 3) Normalize by PSDs
        denominator = psds[cc0][None, :] * psds  # → (P_sum, kk_max)
        rho = cross_psd / denominator

        # 4) Zero out invalid entries
        rho[~mask] = 0

        return rho

    @staticmethod
    def dedupe_modulation(mod_sets, harm_sets, tol=1e-6):
        """
        Remove duplicate modulation sets and remap harmonic sets accordingly.
        """
        mod_sets_unique = []
        map_idx = np.empty(len(mod_sets), dtype=int)

        for i, S in enumerate(mod_sets):
            found = False
            for j, U in enumerate(mod_sets_unique):
                # only compare if they have the same number of elements
                if S.shape == U.shape and np.allclose(S, U, atol=tol):
                    map_idx[i] = j
                    found = True
                    break

            if not found:
                mod_sets_unique.append(S)
                map_idx[i] = len(mod_sets_unique) - 1

        # remap harmonic_sets
        harm_sets_unique = np.full_like(harm_sets, -1)
        mask = harm_sets >= 0
        harm_sets_unique[mask] = map_idx[harm_sets[mask]]

        return mod_sets_unique, harm_sets_unique



    @staticmethod
    def apply_local_coherence_masks(masks: np.ndarray, harmonic_info: 'HarmonicInfo', cov_list: list[np.ndarray],
                                    signals: dict, group_by_set=True) -> ('HarmonicInfo', list, dict):
        """
        assigning np.nan to unused data is useful for debugging but can be safely commented out if it slows down
        the program
        masks: there is one per harmonic set, unless group_by_set=False, then len(masks) = K_nfft
        """

        M, K_nfft = signals['noisy']['stft'].shape[:2]

        if isinstance(cov_list, np.ndarray):
            cov_list = [cov_list]

        for kk in range(K_nfft):
            cc, P = harmonic_info.get_harmonic_set_and_num_shifts(kk)
            mask = masks[cc] if group_by_set else masks[kk]
            P_new = int(np.sum(mask > 0)) // M
            if P_new < P:
                MP_new = int(np.sum(mask > 0))
                squared_mask = np.ix_(mask, mask)

                for idx, cov in enumerate(cov_list):
                    # Remove uncorrelated data from noisy covariance matrix
                    cov[kk, :MP_new, :MP_new] = cov[kk][squared_mask]

                    # useful for debug, can also choose not to overwrite for potential speed gains
                    cov[kk, MP_new:, :] = np.nan
                    cov[kk, :, MP_new:] = np.nan

                    cov_list[idx] = cov

                # Remove uncorrelated data from signals
                for name in signals:
                    if 'mod_stft_3d' in signals[name]:
                        signals[name]['mod_stft_3d'][:, :MP_new, kk] = signals[name]['mod_stft_3d'][:, mask, kk]
                        signals[name]['mod_stft_3d'][:, MP_new:, kk] = np.nan  # useful for debug
                        signals[name]['mod_stft_3d_conj'][:, :MP_new, kk] = signals[name]['mod_stft_3d_conj'][:, mask, kk]
                        signals[name]['mod_stft_3d_conj'][:, MP_new:, kk] = np.nan  # useful for debug

        # Finally, modify harmonic_info to reflect changes
        if group_by_set: # if we group by harmonic sets, we have one mask per set
            num_harmonic_sets = harmonic_info.num_shifts_per_set.size
            for cc in range(num_harmonic_sets):
                # Mask modulations (keep only relevant ones)
                num_mods_og = len(harmonic_info.alpha_mods_sets[cc])  # original number of modulations for the set
                mask_single_ch = masks[cc][::M][:num_mods_og]
                harmonic_info.alpha_mods_sets[cc] = harmonic_info.alpha_mods_sets[cc][mask_single_ch]

        else:  # if we do not group by harmonic sets, we have one mask per frequency bin
            harm_sets = np.arange(K_nfft)
            mod_sets = []
            for kk in range(K_nfft):
                cc, P = harmonic_info.get_harmonic_set_and_num_shifts(kk)
                mask = masks[cc] if group_by_set else masks[kk]
                mods_cc = harmonic_info.alpha_mods_sets[cc]
                mask_single_ch = mask[::M][:mods_cc.size]
                mod_sets.append(mods_cc[mask_single_ch])

            mod_sets, harm_sets = CoherenceManager.dedupe_modulation(mod_sets, harm_sets)
            assert np.allclose(mod_sets[0], 0)  # assume we found the default set at the beginning
            harmonic_info.alpha_mods_sets = mod_sets[1:]

            # Save a copy of the harmonic sets before coherence removal
            harmonic_info.harmonic_sets_before_coh = copy.deepcopy(harmonic_info.harmonic_sets)
            harmonic_info.harmonic_sets = harm_sets - 1

        # Count new number of modulations
        harmonic_info.num_shifts_per_set_before_coh = copy.deepcopy(harmonic_info.num_shifts_per_set)
        harmonic_info.num_shifts_per_set = harmonic_info.get_num_shifts_per_set(harmonic_info.alpha_mods_sets)

        # Update harmonic_bins and mask_harmonic_bins
        harmonic_info.mask_harmonic_bins = harmonic_info.harmonic_sets >= 0
        harmonic_info.harmonic_bins = np.where(harmonic_info.mask_harmonic_bins)[0]

        return harmonic_info, cov_list, signals

    @classmethod
    def remove_uncorrelated_modulations_local_coherence(cls, cov_read: np.ndarray,
                                                        signals: dict, harm_info: 'HarmonicInfo',
                                                        cov_write_list: list[np.ndarray] = None,
                                                        harmonic_threshold=0, debug_plots=False, group_by_set=True):
        """
        There used to be one mask per set (group_by_set=True).
        New version: there is one mask per each frequency bin, potentially (group_by_set=False)
        """

        P_all = harm_info.get_num_shifts_all_frequencies()

        if not cov_write_list:  # if not provided, read and write from/to same covariance matrix
            cov_write_list = [cov_read]

        if np.allclose(P_all, 1.) or harmonic_threshold == 0:  # Nothing to do, no frequency shifts are being applied to the signal
            return *cov_write_list, signals, harm_info

        M, K_nfft = signals['noisy']['stft'].shape[:2]
        min_rel_pow_psd = 0 if group_by_set else 1.e+3  # does not make much difference on synthetic data
        coherence = cls.covariance_to_coherence(cov_read, squared=True, M_times_P_all=M * P_all,
                                                min_relative_power=min_rel_pow_psd)

        if debug_plots:
            for kk in range(K_nfft):
                cls.debug_plots_coherence(kk, M, coherence, harm_info)

        correlation_masks = (
            cls.get_correlation_masks(coherence, harm_info, M, threshold_harmonic=harmonic_threshold,
                                      group_by_set=group_by_set))

        harm_info, cov_write_list, signals = (
            cls.apply_local_coherence_masks(correlation_masks, harm_info, cov_write_list, signals,
                                            group_by_set=group_by_set))

        return *cov_write_list, signals, harm_info

    @classmethod
    def debug_plots_coherence(cls, kk, M, coherence, harm_info):
        harmonic_set_idx, P = harm_info.get_harmonic_set_and_num_shifts(kk)
        MP = M * P
        sel_cov = kk, slice(MP), slice(MP)  # corresponds to [kk, :M*P, :M*P]

        if P > 1:
            center_freqs = harm_info.harmonic_freqs_hz[harmonic_set_idx]
            corresponding_mods = harm_info.alpha_mods_sets[harmonic_set_idx]
            effective_freqs_cov = center_freqs - corresponding_mods
            effective_freqs_cov_str = ', '.join([str(int(x)) for x in effective_freqs_cov])
            u.plot_matrix(coherence[sel_cov], amp_range=(0, 1), log=False, normalized=False,
                          title=f"Bin {kk:.0f} | Freqs {effective_freqs_cov_str}Hz")
            # title=f"Frequencies (Hz): {effective_freqs_cov_str}")
            pass

    @classmethod
    def get_adaptive_harmonic_threshold(cls, harmonic_threshold_, idx_chunk):

        # if idx_chunk <= 20:
        #     return harmonic_threshold_ / 4
        # elif idx_chunk <= 40:
        #     return harmonic_threshold_ / 2
        # elif idx_chunk <= 80:
        #     return harmonic_threshold_ / 1.5

        return harmonic_threshold_

    @staticmethod
    def get_correlation_masks(coherence, harmonic_info, M, threshold_harmonic=0.0,
                              group_by_set=True) -> np.ndarray:
        """ Get masks for modulated components based on coherence matrix.
        One mask per harmonic set (or per frequency bin if group_by_set=False)."""

        if threshold_harmonic < 0 or threshold_harmonic > 1:
            raise AttributeError(f"{threshold_harmonic = }, but it must be a real value between 0 and 1.")

        if not (np.all(coherence >= -0.01) and np.all(coherence <= 1.01)):
            raise ValueError("Input should be real-valued matrix with values in between 0 and 1")

        num_sets = harmonic_info.num_shifts_per_set.size
        if num_sets <= 1:
            warnings.warn(f"There are only {num_sets = }")

        if harmonic_info.harmonic_sets.size <= 1:
            warnings.warn(f"{harmonic_info.harmonic_sets = }")

        MP_max = coherence.shape[1]

        if group_by_set:
            row_mask = np.zeros((num_sets, MP_max), dtype=bool)
            for cc in range(row_mask.shape[0]):
                P = harmonic_info.num_shifts_per_set[cc]
                MP = M * P

                # Select all spectral bins that belong to cc-th harmonic set. They are usually next to each other.
                bins_kk_belonging_to_set_cc = harmonic_info.harmonic_sets == cc

                if np.allclose(bins_kk_belonging_to_set_cc, False):
                    row_mask[cc, :M] = True  # just retain non-modulated components and continue
                    continue

                coherence_cc = coherence[bins_kk_belonging_to_set_cc, :MP, :MP]
                coherence_cc = np.mean(coherence_cc, axis=0)  # average over bins in same harmonic set

                coherence_single_mic, _ = CoherenceManager.block_mean_loop(coherence_cc, M)  # average over microphones
                # coherence_single_mic = coherence_cc[::M, ::M]  # just use reference microphone

                # Selection is based on coherence between non-modulated and modulated components (first row of matrix)
                # We don't care about coherence among modulated components as such! # TODO useless data can be discarded earlier on?
                row_mask_single_mic = coherence_single_mic[0, :] >= threshold_harmonic
                row_mask_single_mic[0] = True  # always keep non-modulated data

                row_mask[cc, :MP] = np.repeat(row_mask_single_mic, M)
        else:
            row_mask = np.zeros((len(harmonic_info.harmonic_sets), MP_max), dtype=bool)
            for kk in range(row_mask.shape[0]):
                cc, P = harmonic_info.get_harmonic_set_and_num_shifts(kk)

                if P == 1:
                    row_mask[kk, :M] = True
                    continue

                MP = M * P
                coherence_kk = coherence[kk, :MP, :MP]

                # coherence_single_mic, _ = CoherenceManager.block_mean_loop(coherence_kk, M)  # average over microphones
                coherence_single_mic = coherence_kk[::M, ::M]  # just use reference microphone

                # Selection is based on coherence between non-modulated and modulated components (first row of matrix)
                # We don't care about coherence among modulated components as such! # TODO useless data can be discarded earlier on?
                row_mask_single_mic = coherence_single_mic[0, :] >= threshold_harmonic
                row_mask_single_mic[0] = True  # always keep non-modulated data

                row_mask[kk, :MP] = np.repeat(row_mask_single_mic, M)

        return row_mask

    @staticmethod
    def block_mean_loop(X, M):
        h, w = X.shape
        assert h % M == 0 and w % M == 0

        H = h // M
        W = w // M
        Y = np.empty((H, W), dtype=X.dtype)
        Y_same_shape = np.empty(X.shape, dtype=X.dtype)

        for i in range(H):
            for j in range(W):
                block = X[i * M:(i + 1) * M, j * M:(j + 1) * M]
                Y_same_shape[i * M:(i + 1) * M, j * M:(j + 1) * M] = np.mean(block)
                Y[i, j] = np.mean(block)

        return Y, Y_same_shape

    @staticmethod
    def covariance_to_coherence(cov_all_freqs, squared=True, M_times_P_all=np.array([]),
                                min_relative_power=0) -> np.ndarray:
        """
        Normalize covariance matrices to correlation-like matrices with ones on the diagonal.

        Parameters
        ----------
        cov_all_freqs : np.ndarray
            Input covariance matrix of shape (K, PM, PM), where K is the number of frequency bins
            and PM is the number of virtual sensors.

        Returns
        -------
        coeff : list of np.ndarray
            List of K complex coefficient matrices with unit diagonal and local_coherence off-diagonals.
        """

        K, PM, _ = cov_all_freqs.shape
        if squared:
            coeff = np.zeros_like(cov_all_freqs, dtype=float)
        else:
            coeff = np.zeros_like(cov_all_freqs, dtype=np.complex128)

        global_max = np.max(np.diagonal(cov_all_freqs, axis1=1, axis2=2).real)

        # TODO: use max_bin information to avoid looping over non-harmonic bins (useless computations)
        for kk in range(K):
            # Select valid elements: covariance allocated based on P_max, but some values are 0 for non-harmonic bins
            sel = slice(None), slice(None)
            if M_times_P_all.size > 0:
                sel = slice(M_times_P_all[kk]), slice(M_times_P_all[kk])

            cov_kk = cov_all_freqs[kk][sel]

            if not np.all(np.isfinite(cov_kk)):
                warnings.warn("Not finite input!")

            if not np.allclose(np.diag(cov_kk).imag, 0):
                raise ValueError("The powers should be real")

            psds = np.diag(cov_kk).real

            if min_relative_power > 0:
                psds = np.maximum(psds, global_max / min_relative_power)  # limit value from below to avoid artefacts when diving by this
                norm_matrix = np.outer(psds, psds).real
                local_coherence = np.abs(cov_kk) ** 2 / norm_matrix
            else:
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    if squared:
                        # Coherence: |S_yx|^2 / (S_y S_x)
                        norm_matrix = np.outer(psds, psds).real
                        local_coherence = np.where(norm_matrix != 0, np.abs(cov_kk) ** 2 / norm_matrix, 0).real
                    else:
                        # Correlation: S_yx / sqrt(S_y S_x)
                        norm_matrix = np.sqrt(np.outer(psds, np.conj(psds)))
                        local_coherence = np.where(norm_matrix != 0, cov_kk / norm_matrix, 0)
                        np.fill_diagonal(local_coherence, 1.0 + 0j)

            coeff[kk][sel] = local_coherence

        if not np.all(np.isfinite(coeff)):
            warnings.warn("Some element in coherence matrix is not finite. Check!")

        if squared and (not (np.all(coeff >= -0.01) and np.all(coeff <= 1.01))):
            warnings.warn("Elements in coherence matrix should be within [0, 1]")

        return coeff
