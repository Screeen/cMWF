import warnings

import numba
import numpy as np
import scipy
from numba import njit, prange
from scipy.signal import ShortTimeFFT

from . import utils as u
from . import globs as g
from .harmonic_info import HarmonicInfo

u.set_printoptions_numpy()
u.set_plot_options()


class Modulator:
    def __init__(self, max_len_samples_, fs_, mods_list, use_filters=False, fast_version=False,
                 max_rel_dist_alpha: float = 1.e-4, max_freq_cyclic_hz=np.inf):
        """

        :param max_len_samples_:
        :param fs_:
        :param mods_list: mods_list is a list of vectors of frequencies to modulate the signals with.
        :param use_filters:
        """

        self.max_rel_dist_alpha = max_rel_dist_alpha
        self.max_freq_cyclic_hz = max_freq_cyclic_hz

        self.check_validity_modulation_list(mods_list)

        # transform it into a long vector, but keep the transformation matrix.
        alpha_vec_hz_ = np.concatenate(mods_list)
        if alpha_vec_hz_.ndim != 1:
            raise ValueError(f"alpha_vec_hz_ should be 1D but {alpha_vec_hz_.shape = }")

        self.alpha_inv = np.array([])
        if fast_version:
            # Compute unique alpha values while preserving structure
            self.alpha_vec_hz_, self.alpha_inv = Modulator.unique_with_relative_tolerance_fast(alpha_vec_hz_,
                                                                                               tol=self.max_rel_dist_alpha,
                                                                                               return_inverse=True)
        else:
            self.alpha_vec_hz_ = alpha_vec_hz_

        self.mod_matrix = Modulator.compute_modulation_matrix(N_num_samples=max_len_samples_,
                                                              fs_=fs_, alpha_vec_hz=self.alpha_vec_hz_)
        self.mod_matrix = np.asfortranarray(self.mod_matrix)

        self.filter_coeffs = []
        if not use_filters:
            return
        self.compute_filter_coefficients(alpha_vec_hz_, fs_)

    @staticmethod
    def check_validity_modulation_list(mods_list):
        if not isinstance(mods_list, list):
            raise AttributeError(f"mods_list must be a list")
        for mods in mods_list:
            if not isinstance(mods, np.ndarray):
                raise AttributeError(f"mods_list must be a list of ndarray")
            if mods.size > 0 and mods[0] != 0:
                raise AttributeError(f"The first element of each modulation vec must be 0, but {mods = }.")

    def compute_filter_coefficients(self, alpha_vec_hz_, fs_):
        # Modulation in the time-domain corresponds to CIRCULAR frequency shifts in the frequency domain.
        # if the frequency axis goes, say from -fs/2 to fs/2, and the spectrum is Gaussian like centered at 0,
        # a positive frequency shift of f0 will shift the spectrum to the right.
        # So the "low negative frequencies" will be shifted to the right and will become "low positive frequencies",
        # but mirrored at the center.
        # From measurements on 17/2/24, however, it seems that the filters I used to fight this issue are not effective.
        for alpha_hz in alpha_vec_hz_:
            if alpha_hz == 0:
                self.filter_coeffs.append((None, None))
                continue
            if alpha_hz > 0:
                # Shifting the spectrum to higher frequencies.
                # Cut high frequencies, so they don't wrap around at the bottom.
                # Cutting low freqs might make sense, as spectrum is mirrored at the center, but it is not effective.
                cutoff_freq = fs_ / 2 - np.abs(alpha_hz) / 2
                b, a = scipy.signal.butter(5, cutoff_freq / (fs_ / 2), btype='lowpass', analog=False, output='ba')
            else:
                # Shifting the spectrum to lower frequencies.
                # Highpass: keep high freqs, cut low frequencies, so they don't wrap around at the top.
                # Cutting high freqs might make sense, as spectrum is mirrored at the center, but it is not effective.
                cutoff_freq = np.abs(alpha_hz) / 2
                b, a = scipy.signal.butter(5, cutoff_freq / (fs_ / 2), btype='highpass', analog=False, output='ba')

            self.filter_coeffs.append((b, a))

    def modulate(self, y):
        # Input signal should have shape: (num_microphones, 1, num_samples)
        # Mod matrix has shape: (num_microphones, num_modulations, num_samples)
        # Output shape is  (M=num_microphones, P=num_modulations, num_samples)

        if not np.all(np.isreal(y)):
            raise ValueError("Input signal should be real")

        if not (y.ndim == 3 and y.shape[1] == 1):
            raise ValueError("Input signal should have shape: (num_microphones, 1, num_samples)")

        # Low- or high-pass filter the signal before modulation
        y = np.broadcast_to(y, (y.shape[0], self.mod_matrix.shape[1], y.shape[-1])).copy(order='F')
        if len(self.filter_coeffs) > 1:
            for idx, (b, a) in enumerate(self.filter_coeffs):
                if b is not None:
                    y[:, idx] = scipy.signal.lfilter(b, a, y[:, idx])

        # Modulate
        y_mod = y * self.mod_matrix[..., :y.shape[-1]]

        return y_mod

    def demodulate(self, y):
        # At least one element should be complex
        if not np.any(np.iscomplex(y)):
            raise ValueError("Input signal should be complex")
        # Input signal should have shape: (num_microphones, num_modulations, num_samples)
        return y * self.mod_matrix[0, :, :y.shape[-1]].conj()
        # return x * self.demod_matrix[0, :, :x.shape[-1]]

    def compute_modulated_signals(self, sig_dict, SFT, P_max, names_signals_to_modulate=None):
        """ Modulate the signals 'names_signals_to_modulate' in the dictionary 'sig_dict' with the modulation frequencies
            contained in 'self.alpha_vec_hz_'. """
        # TODO: much faster if we directly store the stft of the modulated signals in the re-arranged form,
        #  instead of storing the STFTs and then reshaping them.

        P_sum = len(self.alpha_vec_hz_)  # new implementation: this is the sum of all alphas
        M = sig_dict['noisy']['stft'].shape[0]
        L1_frames_all_chunks = sig_dict['noisy']['stft'].shape[-1]
        assert np.isclose(np.log2(SFT.mfft), int(np.log2(SFT.mfft)))
        nfft_real = SFT.mfft // 2 + 1

        if names_signals_to_modulate is None:
            warnings.warn("No signals to modulate are provided. Modulating all signals.")
            names_signals_to_modulate = sig_dict.keys()
        assert all([signal_name in sig_dict.keys() for signal_name in names_signals_to_modulate])

        alpha_rep, alpha_unique = self.check_repeated_modulating_frequencies(self.alpha_vec_hz_,
                                                                             max_rel_dist=self.max_rel_dist_alpha)

        for signal_name in names_signals_to_modulate:
            num_frames = L1_frames_all_chunks if signal_name != 'noise_cov_est' else \
                sig_dict['noise_cov_est']['stft'].shape[-1]

            if P_sum <= 1 and P_max is not None:  # only for recursive implementation
                # Notice that the STFT is computed in 'data_generator.generate_noise_noisy' and
                # 'data_generator.generate_target_signal'.
                # Reshape the signals following P_max, but pad with zeros if P_max > P_sum
                # Option 1: All zeros are in place of modulated components.
                # Can cause sudden jump of volume when fundamental frequency is 0.
                # Diving the signals by P_max will make the jump less noticeable.

                sig_dict[signal_name]['mod'] = np.zeros((M, P_max, sig_dict[signal_name]['time'].shape[-1]),
                                                        dtype=np.complex128, order='F')
                sig_dict[signal_name]['mod'][:, :P_sum, :] = sig_dict[signal_name]['time'][:, np.newaxis]

                sig_dict[signal_name]['mod_stft'] = np.zeros((M, P_max, nfft_real, num_frames), dtype=np.complex128,
                                                             order='F')
                sig_dict[signal_name]['mod_stft'][:, :P_sum, :, :] = sig_dict[signal_name]['stft'][:, np.newaxis]

                # Option 2: Copy the same data to all the P_max modulations. A bit strange, but it works.
                # shape_mod_time = (M, P_max, sig_dict[signal_name]['time'].shape[-1])
                # sig_dict[signal_name]['mod'] = np.broadcast_to(sig_dict[signal_name]['time'][:, np.newaxis, :],
                #                                                shape_mod_time).copy()
                # shape_mod_stft = (M, P_max, stft_obj.mfft, num_frames)
                # sig_dict[signal_name]['mod_stft'] = np.broadcast_to(sig_dict[signal_name]['stft'][:, np.newaxis],
                #                                                     shape_mod_stft).copy()
            else:
                # shape_mod_time: (M, P_sum, num_samples)
                sig_dict[signal_name]['mod'] = self.modulate(sig_dict[signal_name]['time'][:, np.newaxis, :])

                # Compute stft only for indices whose alpha_rep[idx] == -1 and copy the stft to the rest.
                sig_dict[signal_name]['mod_stft'] = np.zeros((M, P_sum, nfft_real, num_frames), dtype=np.complex128,
                                                             order='F')
                for idx, alpha in enumerate(self.alpha_vec_hz_):
                    if alpha_rep[idx] == -1:
                        sig_dict[signal_name]['mod_stft'][:, idx] = SFT.stft(
                            sig_dict[signal_name]['mod'][:, idx])[:, :nfft_real]
                    else:
                        sig_dict[signal_name]['mod_stft'][:, idx] = sig_dict[signal_name]['mod_stft'][:, alpha_rep[idx]]

                # shape_mod_stft: (M, P_sum, mfft, num_frames)
                # sig_dict[signal_name]['mod_stft'] = stft_obj.stft(sig_dict[signal_name]['mod'])[..., :num_frames]

        # Check that the modulated signals are correct
        # for signal_name in res.keys():
        #     if not np.allclose(res[signal_name]['mod_stft'][:, 0], res[signal_name]['stft']):
        #         warnings.warn(f"Modulated signal {signal_name} is not correct.")

        return sig_dict

    @staticmethod
    def rearrange_modulated_signals(sig_dict, names_signals_to_modulate, num_shifts_per_set, P_max_single_harmonic):
        """ Rearrange the modulated signals in the STFT domain to arrays, combining the M and the P channels. """

        if any([np.any((num_shifts_single_set <= 0) | (num_shifts_single_set > P_max_single_harmonic)) for
                num_shifts_single_set in num_shifts_per_set]):
            raise AttributeError(
                f"{num_shifts_per_set = } should be strictly larger than 0 and smaller than, or equal to {P_max_single_harmonic = }")

        for signal_name in names_signals_to_modulate:
            M, P_max, nfft_real, num_frames = sig_dict[signal_name]['mod_stft'].shape

            # assert np.sum(num_shifts_per_set) == P_max  # temporarily
            num_harmonic_sets = len(num_shifts_per_set)

            # # First dimension of 3d signal corresponds to M * P_max
            # sig_dict[signal_name]['mod_stft_3d'] = np.reshape(sig_dict[signal_name]['mod_stft'],
            #                                                   (-1, mfft, num_frames), order='F')
            # sig_dict[signal_name]['mod_stft_3d_conj'] = sig_dict[signal_name]['mod_stft_3d'].conj()

            # This will not be actually 3d, we just keep the name for convenience for now.
            shifts_per_set = np.cumsum(num_shifts_per_set)
            sig_dict[signal_name]['mod_stft_3d'] = np.zeros(
                (num_harmonic_sets, M * P_max_single_harmonic, nfft_real, num_frames),
                dtype=np.complex128, order='F')

            for hh in range(num_harmonic_sets):
                sel = slice(None), slice(shifts_per_set[hh] - num_shifts_per_set[hh], shifts_per_set[hh])
                sig_dict[signal_name]['mod_stft_3d'][hh, :M * num_shifts_per_set[hh]] = (
                    np.reshape(sig_dict[signal_name]['mod_stft'][sel], (-1, nfft_real, num_frames), order='F'))

            sig_dict[signal_name]['mod_stft_3d_conj'] = np.conj(sig_dict[signal_name]['mod_stft_3d'])

        return sig_dict

    def compute_reshaped_modulated_signals(self, sig_dict, SFT: ShortTimeFFT, P_max_cfg=1,
                                           names_signals_to_modulate=None,
                                           num_shifts_per_set=None, name_input_sig='noisy'):
        """
        Modulate the signals in the dictionary 'sig_dict' with the modulation frequencies
        contained in 'self.alpha_vec_hz_', and store them directly in the re-arranged form.

        - P_max_cfg could actually be calculated as the maximum among num_shifts_per_set, unless we are using recursive implementation,
        where the number of harmonics can change at every chunk/frame but there is a maximum allocated P_max.

        The difficulty is that each modulation is applied M times.
        The reshape from M, P to M*P should be done early when the data is not large (i.e. before the modulation, ideally,
        or at least before STFT). Right now is done before STFT, which should be fine.

        sig (M, T)
        sig_mod (M, P_unique, T)
        sig_mod_flat (M*P_unique, T)
        sig_mod_stft (M*P_unique, K, L)
        sig_mod_stft_conj (M*P_unique, K, L)
        sig_mod_stft_3d (num_harm_sets, P_max, K, L)
        """

        if np.any(num_shifts_per_set == 0):
            print(f"{num_shifts_per_set = }. No modulations will be performed.")

        if names_signals_to_modulate is None:
            warnings.warn("No signals to modulate are provided. Modulating all signals.")
            names_signals_to_modulate = sig_dict.keys()

        assert all([signal_name in sig_dict.keys() for signal_name in names_signals_to_modulate])
        assert all([num_shifts <= P_max_cfg for num_shifts in num_shifts_per_set])

        M = sig_dict[name_input_sig]['stft'].shape[0]
        nfft_real = SFT.mfft // 2 + 1
        P_sum = len(self.alpha_vec_hz_)  # new implementation: this is the length of the concatenation of all alphas
        num_harmonic_sets = len(num_shifts_per_set)

        for signal_name in names_signals_to_modulate:
            # Initialize the final data structure with the required ordering
            num_frames = sig_dict[signal_name]['stft'].shape[-1]
            sig_dict[signal_name]['mod_stft_3d'] = np.zeros(
                (num_harmonic_sets, M * P_max_cfg, nfft_real, num_frames),
                dtype=np.complex128, order='F')
            sig_dict[signal_name]['mod_stft_3d_conj'] = np.zeros(
                (num_harmonic_sets, M * P_max_cfg, nfft_real, num_frames),
                dtype=np.complex128, order='F')

            # No harmonicity detected
            if P_sum <= 1 and P_max_cfg is not None:
                sig_dict[signal_name]['mod_stft_3d'][0, :M] = sig_dict[signal_name]['stft']
                sig_dict[signal_name]['mod_stft_3d_conj'][0, :M] = sig_dict[signal_name]['stft_conj']
                continue

            # Modulate data, convert it to STFT, store it in the correct order
            sig_dict[signal_name] = self.compute_modulated_data_stft(sig_dict[signal_name], SFT,
                                                                     num_shifts_per_set)

        return sig_dict

    def compute_modulated_data_stft(self, signal, SFT, num_shifts_per_set):
        # Transform to STFT domain and compute conjugate
        # TODO not all computed frequencies, we could only compute the required ones to save on computation

        nfft_real = SFT.mfft // 2 + 1

        max_freq_bins = nfft_real
        if np.isfinite(self.max_freq_cyclic_hz):
            max_freq_bins = int(np.ceil(self.max_freq_cyclic_hz * SFT.delta_f))

        # Modulate the data in the time domain
        modulated_data = self.modulate(signal['time'][:, np.newaxis, :])

        # Convert to STFT domain
        # mod_data_stft.shape = (Mics, P_sum, K_nfft, L_frames)
        mod_data_stft = SFT.stft(modulated_data)[:, :, :max_freq_bins, :signal['stft'].shape[-1]]
        # mod_data_stft_conj = np.conj(mod_data_stft)

        if self.alpha_inv.size == 0:
            raise ValueError("Remember to initialize with fast_version=True to compute self.alpha_inv")

        if np.any(self.alpha_inv >= self.alpha_vec_hz_.size):
            raise IndexError(f"{self.alpha_inv = } out of bounds for unique_data.")

        # signal = self.rearrange_modulated_signals_new(signal, mod_data_stft, mod_data_stft_conj,
        #                                                    num_shifts_per_set,
        #                                                    self.alpha_inv)

        # Should be identical to version above (but no test implemented!) with some tricks to optimize for speed
        # signal = self.rearrange_modulated_signals_new_fast_4(signal, mod_data_stft, mod_data_stft_conj,
        #                                                      num_shifts_per_set, self.alpha_inv)

        signal = self.rearrange_modulated_signals_new_fast_4_no_conj(signal, mod_data_stft,
                                                                     num_shifts_per_set, self.alpha_inv)

        return signal

    @staticmethod
    def rearrange_modulated_signals_new(signal, mod_data_stft, mod_data_stft_conj, num_shifts_per_set, alpha_inv):

        # Define shift structure: assume that num_shifts_per_set[0] modulations in self.alpha_vec_hz_ should be applied
        # to first harmonic set, num_shifts_per_set[1] modulations applied to second set, and so on.
        # The order follows that of alpha_mods_list when it was provided as input to Modulator (before sorting).
        sets_idx_lists = Modulator.split_into_sequential_sublists(num_shifts_per_set)

        # Iterate over sets and shifts while keeping correct sensor ordering
        num_sets = len(num_shifts_per_set)
        M = signal['stft'].shape[0]

        for ss in range(num_sets):  # Per each harmonic set
            aa_mm = 0  # Tracks the correct position in data across sensors
            for aa in range(num_shifts_per_set[ss]):  # Per each modulation in the harmonic set

                # Original index of modulation in alpha_mods given as input to Modulator.
                # This is before sorting and selecting unique values!
                alpha_idx_og = sets_idx_lists[ss][aa]
                alpha_idx_sorted = alpha_inv[alpha_idx_og]  # Index in unique alpha values
                # debug_alpha_hz = self.alpha_vec_hz_[self.alpha_inv[alpha_idx_og]]  # check what mod we processing

                # Assign data and store mapping
                for mm in range(M):  # Per each microphone (modulation is the same, data is different due to RIR effect)
                    # Shape: (num_harmonic_sets, M * P_max_cfg, nfft_real, num_frames),
                    signal['mod_stft_3d'][ss, aa_mm] = mod_data_stft[mm, alpha_idx_sorted]
                    signal['mod_stft_3d_conj'][ss, aa_mm] = mod_data_stft_conj[mm, alpha_idx_sorted]

                    aa_mm += 1  # Move to the next sensor slot

        return signal

    @staticmethod
    @njit(cache=True, parallel=True)
    def _copy_blocks(split_rows, flat_c, flat_cc, out, outc):
        # parallelize over ss if you like
        for ss in prange(split_rows.shape[0] - 1):
            r0 = split_rows[ss]
            r1 = split_rows[ss + 1]
            out[ss, : (r1 - r0), :, :] = flat_c[r0:r1]
            outc[ss, : (r1 - r0), :, :] = flat_cc[r0:r1]

    @staticmethod
    @njit(cache=True, parallel=True)
    def _copy_blocks_no_conj(split_rows, flat_c, out):
        # parallelize over ss if you like
        for ss in prange(split_rows.shape[0] - 1):
            r0 = split_rows[ss]
            r1 = split_rows[ss + 1]
            out[ss, : (r1 - r0), :, :] = flat_c[r0:r1]

    @staticmethod
    def rearrange_modulated_signals_new_fast_4_no_conj(
            signal,
            mod_data_stft,  # (M, P, nfft, T)
            num_shifts_per_set,
            alpha_inv):

        M, P, nfft, T = mod_data_stft.shape

        # one big sort → contiguous (len(alpha_inv), M, nfft, T)
        mod_c_sorted = np.moveaxis(mod_data_stft, 1, 0)[alpha_inv]

        # collapse first two dims → (P'*M, nfft, T)
        flat_c = mod_c_sorted.reshape(-1, nfft, T, order='C')

        # build row-splits in units of M
        row_counts = np.array(num_shifts_per_set, dtype=np.int64) * M
        split_rows = np.empty(row_counts.size + 1, dtype=np.int64)
        split_rows[0] = 0
        for i in range(row_counts.size):
            split_rows[i + 1] = split_rows[i] + row_counts[i]

        # delegate the per‐set copy to Numba
        Modulator._copy_blocks_no_conj(split_rows, flat_c,
                                       signal['mod_stft_3d'])

        signal['mod_stft_3d_conj'] = np.conj(signal['mod_stft_3d'])

        return signal

    @staticmethod
    def rearrange_modulated_signals_new_fast_4(
            signal,
            mod_data_stft,  # (M, P, nfft, T)
            mod_data_stft_conj,
            num_shifts_per_set,
            alpha_inv):

        M, P, nfft, T = mod_data_stft.shape

        # one big sort → contiguous (len(alpha_inv), M, nfft, T)
        mod_c_sorted = np.moveaxis(mod_data_stft, 1, 0)[alpha_inv]
        mod_cc_sorted = np.moveaxis(mod_data_stft_conj, 1, 0)[alpha_inv]

        # collapse first two dims → (P'*M, nfft, T)
        flat_c = mod_c_sorted.reshape(-1, nfft, T, order='C')
        flat_cc = mod_cc_sorted.reshape(-1, nfft, T, order='C')

        # build row-splits in units of M
        row_counts = np.array(num_shifts_per_set, dtype=np.int64) * M
        split_rows = np.empty(row_counts.size + 1, dtype=np.int64)
        split_rows[0] = 0
        for i in range(row_counts.size):
            split_rows[i + 1] = split_rows[i] + row_counts[i]

        # delegate the per‐set copy to Numba
        Modulator._copy_blocks(split_rows, flat_c, flat_cc,
                               signal['mod_stft_3d'],
                               signal['mod_stft_3d_conj'])

        return signal

    @staticmethod
    def rearrange_modulated_signals_new_fast_3(signal,
                                               mod_data_stft,  # (M, P, nfft, T)
                                               mod_data_stft_conj,
                                               num_shifts_per_set,
                                               alpha_inv):

        M, P, nfft, T = mod_data_stft.shape

        # 1) bring P → front, then one big fancy‐index pass
        mod_c_sorted = np.moveaxis(mod_data_stft, 1, 0)[alpha_inv]  # (len(alpha_inv), M, nfft, T)
        mod_cc_sorted = np.moveaxis(mod_data_stft_conj, 1, 0)[alpha_inv]

        # 2) collapse the first two dims into ‘rows’
        flat_c = mod_c_sorted.reshape(-1, nfft, T, order='C')
        flat_cc = mod_cc_sorted.reshape(-1, nfft, T, order='C')

        # 3) build row‐based splits
        row_counts = [p * M for p in num_shifts_per_set]
        split_rows = np.cumsum([0] + row_counts)

        # 4) single memcpy per set
        for ss in range(len(num_shifts_per_set)):
            r0, r1 = split_rows[ss], split_rows[ss + 1]
            signal['mod_stft_3d'][ss, : (r1 - r0)] = flat_c[r0:r1]
            signal['mod_stft_3d_conj'][ss, : (r1 - r0)] = flat_cc[r0:r1]

        return signal

    @staticmethod
    def rearrange_modulated_signals_new_fast_2(signal,
                                               mod_data_stft,  # (M, P, nfft, T)
                                               mod_data_stft_conj,
                                               num_shifts_per_set,
                                               alpha_inv):

        M, P, nfft, T = mod_data_stft.shape
        mod_c = np.moveaxis(mod_data_stft, 1, 0)  # (P, M, nfft, T)
        mod_cc = np.moveaxis(mod_data_stft_conj, 1, 0)  # (P, M, nfft, T)

        sets_idx_lists = Modulator.split_into_sequential_sublists(num_shifts_per_set)

        for ss, idxs in enumerate(sets_idx_lists):
            sorted_idxs = alpha_inv[idxs]  # this is what breaks contiguity

            # Apply indexing per modulation index
            block = mod_c[sorted_idxs]  # (P_ss, M, nfft, T)
            block_conj = mod_cc[sorted_idxs]

            # Now reshape to collapse (P_ss, M) → (P_ss*M)
            block = block.reshape(-1, nfft, T, order='C')
            block_conj = block_conj.reshape(-1, nfft, T, order='C')

            # One big copy into Fortran array
            signal['mod_stft_3d'][ss, :block.shape[0]] = block
            signal['mod_stft_3d_conj'][ss, :block.shape[0]] = block_conj

        return signal

    @staticmethod
    def rearrange_modulated_signals_new_fast(signal,
                                             mod_data_stft,  # (M, P, nfft, T)
                                             mod_data_stft_conj,
                                             num_shifts_per_set,
                                             alpha_inv):

        sets_idx_lists = Modulator.split_into_sequential_sublists(num_shifts_per_set)
        num_sets = len(num_shifts_per_set)
        M, P, nfft, T = mod_data_stft.shape

        # Bring alpha dim to front and make Fortran‐contiguous (so that later slicing is cheap)
        mod_f = np.asfortranarray(np.moveaxis(mod_data_stft, 1, 0))  # (P, M, nfft, T), F-order
        mod_fc = np.asfortranarray(np.moveaxis(mod_data_stft_conj, 1, 0))

        for ss in range(num_sets):
            orig_alpha_idxs = sets_idx_lists[ss]  # e.g. length P_ss
            sorted_idxs = alpha_inv[orig_alpha_idxs]  # same length

            # slice out one big (P_ss, M, nfft, T) block — still F-contiguous
            block = mod_f[sorted_idxs, :, :, :]
            block_conj = mod_fc[sorted_idxs, :, :, :]

            # collapse the first two dims **in C‐order** so that
            # new_index = aa*M + mm  (exactly what your loops did)
            N1 = block.shape[0] * block.shape[1]
            block = block.reshape((N1, nfft, T), order='C')
            block_conj = block_conj.reshape((N1, nfft, T), order='C')

            # now one big vectorized write per set
            signal['mod_stft_3d'][ss, :N1] = block
            signal['mod_stft_3d_conj'][ss, :N1] = block_conj

        return signal

    def modulate_impulse_response(self, speech_ir, nfft_):

        P = len(self.alpha_vec_hz_)
        M = speech_ir.shape[0]

        # Modulate the impulse response
        # speech_rtf_mod.shape = (M, num_modulations, dft_props['nfft'])
        speech_ir_mod = self.modulate(speech_ir[:, np.newaxis, :])

        # scale doesn't matter here as we divide by the reference mic!
        speech_atf_mod = np.fft.fft(speech_ir_mod, n=nfft_, axis=-1)
        speech_rtf_mod = speech_atf_mod / speech_atf_mod[g.mic0_idx]

        # Form the constraint matrix C. C is a block-matrix. Has P*M rows and P columns.
        # Each column contains a transfer function for the corresponding modulation and zeros for the rest.
        C_rtf = np.zeros((M * P, nfft_, P), dtype=np.complex128, order='F')
        for kk in range(nfft_):
            for mm in range(P):
                mm_M = slice(M * mm, M * (mm + 1))
                C_rtf[mm_M, kk, mm] = speech_rtf_mod[:, mm, kk]

        return C_rtf

    @staticmethod
    def modulate_early_ref(wet_rank1_mod_stft_3d_in, C_rtf, slice_frames):
        """ Add signals that don't change with the modulation to the modulated signals dictionary.
            Plus, compute the rank-1 model based modulated wet signal. """

        M_times_P, nfft, L1_frames_all_chunks = wet_rank1_mod_stft_3d_in.shape
        P = C_rtf.shape[-1]
        M = M_times_P // P

        wet_rank1_mod_stft_3d_out = np.zeros((M_times_P, nfft,), dtype=np.complex128)
        for kk in range(nfft):
            sel = slice(None), kk, slice_frames
            wet_rank1_mod_stft_3d_out[sel] = (C_rtf[kk] @ wet_rank1_mod_stft_3d_in[::M, kk, slice_frames])

        return wet_rank1_mod_stft_3d_out

    @staticmethod
    @numba.jit(cache=True, nopython=True)
    def compute_modulation_matrix(N_num_samples: int, alpha_vec_hz: np.ndarray, fs_: float) -> np.ndarray:
        # Compute the matrix used to shifts signals to the frequencies contained in alpha_vec_hz

        # alpha_vec_hz is 1d: (num_alphas,)
        exponent_without_alpha = 2j * np.pi * np.arange(N_num_samples) / fs_  # size: (N_num_samples,)
        alpha_matrix = np.exp(np.outer(alpha_vec_hz, exponent_without_alpha))  # size: (num_alphas, N_num_samples)

        # Add "num_realizations" dimension to alpha_matrix to get shape to get shape
        # (num_realizations, num_alphas, N_num_samples)
        return alpha_matrix[np.newaxis, ...]

    @staticmethod
    @numba.jit(cache=True, nopython=True)
    def modulate_signal_all_alphas_vec_2d(y: np.ndarray, alpha_vec_hz: np.ndarray, fs_: float) -> np.ndarray:
        # Shifts the signal y to higher frequencies contained in alpha_vec_hz

        # y is 2d: (num_realizations, N_num_samples)
        # alpha_vec_hz is 1d: (num_alphas,)
        N_num_samples = y.shape[-1]
        exponent_without_alpha = 2j * np.pi * np.arange(N_num_samples) / fs_  # size: (N_num_samples,)
        alpha_matrix = np.exp(np.outer(alpha_vec_hz, exponent_without_alpha))  # size: (num_alphas, N_num_samples)

        # Add "alphas" dimension to y and "num_realizations" dimension to alpha_matrix to get shape to get shape
        # (num_realizations, num_alphas, N_num_samples)
        y_alpha_time = y[:, np.newaxis] * alpha_matrix[np.newaxis, ...]

        return y_alpha_time

    def check_repeated_modulating_frequencies(self, alpha_vec_hz_, max_rel_dist=1e-4):
        """
        self.alpha_vec_hz can contain repeated values. This is not a problem, but it is not efficient.
        to save on the number of STFTs, we can compute the STFT of the modulated signals only once.

        Make a vec like this: if alpha_vec = [0, 100, 200, 0, 100, 0], then
        alpha_rep = [-1, -1, -1, 0, 1, 0], where -1 means 'first time this alpha is seen'.

        then we copy (by reference) the STFT of the first time we see the alpha to the other places.
        Get the first occurrence of each alpha in the alpha_vec_hz_. If there are repetitions, the first occurrence
        is taken as the reference for the STFT.
        """
        alpha_rep = np.ones(alpha_vec_hz_.size, dtype=int) * -1
        alpha_unique = []

        for idx, alpha in enumerate(alpha_vec_hz_):
            relative_differences = np.abs(alpha_vec_hz_ - alpha) / (1.e-12 + np.abs(alpha))
            relative_differences[relative_differences < max_rel_dist] = 0
            first_occurrence_mod_freq = np.argmin(relative_differences)
            if not first_occurrence_mod_freq == idx:
                alpha_rep[idx] = first_occurrence_mod_freq
            else:
                alpha_unique.append(alpha)

        return alpha_rep, np.sort(alpha_unique)

    @staticmethod
    def split_into_sequential_sublists(lengths):
        """Converts an array of segment lengths into a list of sequential sublists."""
        start = 0
        result = []
        for length in lengths:
            result.append(list(range(start, start + length)))
            start += length
        return result

    @staticmethod
    def unique_with_relative_tolerance(x, tol=1e-2, return_inverse=False):
        """
        Works like np.unique for floating point numbers, but treats as equal two values that are within tolerance:
        Return modified input array such that values that are relatively close (with relative difference < tol)
        are replaced by the first of such values.
        Uses np.unique to get the unique values and inverse indices.

                Pseudocode:
        Say, alpha_vec = [0, 100, 0, -100, 100].
        When we compute modulations, the vector should be
        alpha_unique = [-100, 0, 100].
        unique_indices = [3, 0, 1] (sorted)
        unique_inverse = [1, 2, 1, 0, 2] (reconstructs alpha_vec from alpha_unique)

        Parameters:
            x (np.ndarray): Input array.
            tol (float): Relative tolerance threshold for considering values close.

        Returns:
            np.ndarray: The unique values after merging close values.
            np.ndarray: The inverse indices mapping each element to its unique value.
        """
        arr_mod = x.copy()
        arr_mod_abs = np.abs(arr_mod)

        # Iterate over the array to find clusters
        for ii in range(len(arr_mod)):
            if np.isclose(arr_mod[ii], 0):
                continue
            for jj in range(ii + 1, len(arr_mod)):
                if np.isclose(arr_mod[jj], 0):
                    continue
                rel_diff = np.abs(arr_mod[ii] - arr_mod[jj]) / (np.maximum(arr_mod_abs[ii], arr_mod_abs[jj]))
                if rel_diff < tol:
                    arr_mod[jj] = arr_mod[ii]

        # Now use np.unique to get the unique values
        return np.unique(arr_mod, return_inverse=return_inverse)

    @staticmethod
    def unique_with_relative_tolerance_fast(x, tol=1e-2, return_inverse=False):
        """
        Works like np.unique for floating point numbers, but treats as equal two values
        whose relative difference < tol.  Values within tol of zero are all merged into 0.
        The representative of each cluster is the first occurrence in the original array.

        Parameters:
            x (np.ndarray): Input array.
            tol (float): Relative tolerance threshold.
            return_inverse (bool): If True, also return inverse indices.

        Returns:
            np.ndarray: The sorted unique values after merging close values.
            np.ndarray (optional): Inverse indices mapping each element to its unique value.
        """
        x = np.asarray(x)
        n = x.size
        if n == 0:
            if return_inverse:
                return np.array([], dtype=x.dtype), np.array([], dtype=int)
            return np.array([], dtype=x.dtype)

        # sort for adjacent comparisons
        order = np.argsort(x)
        inv_order = np.argsort(order)
        x_s = x[order]
        abs_s = np.abs(x_s)

        # relative diffs of neighbors (suppress invalid/div by zero warnings)
        with np.errstate(invalid='ignore', divide='ignore'):
            diffs = np.abs(np.diff(x_s)) / np.maximum(abs_s[:-1], abs_s[1:])
        zeros = np.isclose(x_s, 0)

        # mark close pairs: either both zero or relative‐diff < tol (excluding zero pairs twice)
        close = ((diffs < tol) & (~zeros[:-1]) & (~zeros[1:])) | (zeros[:-1] & zeros[1:])

        # build cluster labels
        cluster = np.empty(n, dtype=int)
        cid = 0
        cluster[0] = 0
        for i in range(1, n):
            if close[i - 1]:
                cluster[i] = cid
            else:
                cid += 1
                cluster[i] = cid
        n_clusters = cid + 1

        # for each cluster, pick the rep as the lowest original index
        rep_idx = np.full(n_clusters, fill_value=n, dtype=int)
        for i in range(n):
            c = cluster[i]
            orig = order[i]
            if orig < rep_idx[c]:
                rep_idx[c] = orig
        rep_vals = x[rep_idx]

        # sorted unique representatives
        reps = np.sort(rep_vals)

        if return_inverse:
            # map each original element to its rep, then to index in sorted reps
            mapped = rep_vals[cluster[inv_order]]
            inv = np.searchsorted(reps, mapped)
            return reps, inv

        return reps

    @staticmethod
    def get_max_len_modulated_signals(signals_unproc, signals_to_modulate):
        # check each time signal and get the maximum length
        return max(
            [signals_unproc[key]['time'].shape[-1] for key in signals_unproc.keys() if key in signals_to_modulate])

    @classmethod
    def modulate_signals(cls, signals_unproc, signals_to_modulate, SFT: ShortTimeFFT, alpha_mods_list, P_max,
                         name_input_sig='noisy'):
        # TODO: the modulated signals are computed for the whole duration of the signals. This is not necessary.

        # def print_log_modulation():
        #     print(f"Modulate at {idx_chunk + 1}/{num_chunks}. {mod_amount}, {alpha_mods_list[0][:2]}")
        # print_log_modulation()

        if np.sum([x in signals_unproc.keys() for x in signals_to_modulate]) == 0:
            warnings.warn(f"{signals_unproc.keys() = }, {signals_to_modulate = }")
            return signals_unproc

        max_len = cls.get_max_len_modulated_signals(signals_unproc, signals_to_modulate)

        mod = Modulator(max_len, SFT.fs, alpha_mods_list, use_filters=False, fast_version=True)

        num_shifts_per_set = HarmonicInfo.get_num_shifts_per_set(alpha_mods_list)
        signals_unproc = mod.compute_reshaped_modulated_signals(signals_unproc, SFT, P_max,
                                                                signals_to_modulate,
                                                                num_shifts_per_set,
                                                                name_input_sig)

        # Commented out as cLCMV is not in use
        # mod_rtf = modulator.Modulator(max_len_ir_atf, fs, mods_list=[[0]], use_filters=use_filters)
        # C_rtf = mod_rtf.modulate_impulse_response(target_ir[:, :max_len_ir_atf], dft_props['nfft'])

        return signals_unproc
