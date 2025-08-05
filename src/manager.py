import copy
import warnings
import numpy as np
import scipy

from . import utils as u
from . import plotter as pl
from .globs import rng


# noinspection PyTupleAssignmentBalance
class Manager:

    def __init__(self):
        self.cached_target_samples = {}
        self.cached_f0 = {}

    @staticmethod
    def choose_loud_bins(S_pow, power_ratio_quiet_definition=1000.):
        # Return indices of bins that are loud -- i.e., have power such that:
        #   S_pow > np.max(S_pow) / power_ratio_quiet_definition
        # power_ratio_quiet_definition: number of times less power than the loudest bin to be considered quiet
        # Higher power_ratio_quiet_definition means more bins are considered loud

        loud_bins = np.where(S_pow > np.max(S_pow) / power_ratio_quiet_definition)[0]
        return loud_bins

    @staticmethod
    def add_noise_snr(clean, snr_db=160, fs=None, noise=np.zeros(()), lp_filtered_noise=False):
        # The noise is scaled to achieve the desired SNR and added to the clean signal.
        # Clean input will never be modified. Output will be a new array.
        # If noise is not provided, it is generated as white Gaussian noise.
        # return noisy, noise

        if snr_db > 150:
            return clean, np.zeros_like(clean)

        if not noise.any():
            if np.isrealobj(clean):
                noise = rng.normal(size=clean.shape)
            else:
                noise = u.circular_gaussian(shape=clean.shape)
        else:
            assert noise.shape == clean.shape, f"Noise and clean signal must have the same shape, but {noise.shape = }" \
                                               f" and {clean.shape = }."

        if lp_filtered_noise:
            cutoff_freq = 1000
            b, a = scipy.signal.butter(4, cutoff_freq / (fs / 2), btype='lowpass', analog=False, output='ba')
            noise = scipy.signal.lfilter(b, a, noise)

        noise_rescaled, _ = Manager.rescale_noise_to_snr(clean, noise, snr_db)
        noisy = clean + noise_rescaled

        return noisy, noise_rescaled

    @staticmethod
    def rescale_noise_to_snr(clean, noise, snr_db):
        """ Rescale the noise to achieve the desired SNR. Return rescaled noise and the scaling coefficient. """

        noise_pow = np.sum(np.abs(noise) ** 2) / noise.size
        sig_pow = np.sum(np.abs(clean) ** 2) / clean.size
        if np.isclose(sig_pow, 0):
            raise ValueError("Signal power is too small. Select a different segment.")

        coeff = np.sqrt((sig_pow / noise_pow) * 10 ** (-snr_db / 10))
        noise = coeff * noise

        # Noise pow after rescaling, to be sure SNR is as desired
        # noise_pow = np.sum(np.abs(noise) ** 2) / noise.size
        # snr_ = 10 * np.log10(sig_pow / noise_pow)
        # print(f"{snr_ = :.2f} dB")
        # assert np.isclose(snr_, snr_db, atol=0.1)

        return noise, coeff

    @staticmethod
    def compute_spectral_and_cyclic_resolutions(fs, nfft, names_scf_estimators, alphas_dirichlet, N_num_samples=None):
        """ Compute spectral and cyclic resolutions based on the SCF estimators that are used. """

        delta_f = fs / nfft
        delta_alpha_dict = dict()

        if names_scf_estimators:
            for name_scf_estimator in names_scf_estimators:
                if name_scf_estimator == 'psd' or name_scf_estimator == 'sample_cov':
                    delta_alpha_dict[name_scf_estimator] = delta_f
                elif name_scf_estimator == 'dirichlet' or name_scf_estimator == 'acp':
                    delta_alpha_dict[name_scf_estimator] = alphas_dirichlet[1] - alphas_dirichlet[0]

        if N_num_samples:
            delta_alpha_min = fs / N_num_samples  # maximum possible resolution (minimum possible delta_alpha)
            for delta_alpha in delta_alpha_dict.values():
                if delta_alpha < delta_alpha_min:
                    raise ValueError(f"{delta_alpha = } is too small: it should be at least {delta_alpha_min = }.")

        return delta_f, delta_alpha_dict

    @staticmethod
    def get_cov_estimation_frames(slice_frames_list, voiced_flag_list=None, min_length_slice=5, overlap_frames=2):
        """ Given a list of tuples (start_frame, end_frame) and a list of voiced flags, return a list of tuples
        (start_sample, end_sample) for each chunk. For covariance estimation, one needs at least min_frames frames.
        if chunk is shorter than min_frames, and voiced_flag is False, then chunk is extended up to min_frames.
        Also, both for voiced and unvoiced chunks, it is good to have some overlap with the previous and next chunk.
        """

        # If not provided, do not extend any chunk, only make them overlap
        if voiced_flag_list is None:
            voiced_flag_list = [True] * len(slice_frames_list)

        # Make chunks overlap
        frames_list_cov_estimation = []
        for slice_frame in slice_frames_list:
            start_frame, end_frame = slice_frame
            start_frame = max(0, start_frame - overlap_frames)
            frames_list_cov_estimation.append((start_frame, end_frame))

        """
        # Extend UNVOICED chunks that are too short
        if min_length_slice > 0:
            frames_list_cov_estimation_new = []
            for slice_frame, voiced_flag in zip(frames_list_cov_estimation, voiced_flag_list):
                start_frame, end_frame = slice_frame
                if end_frame - start_frame + 1 < min_length_slice and not voiced_flag:
                    start_frame = max(0, start_frame - min_length_slice)
                frames_list_cov_estimation_new.append((start_frame, end_frame))
            return frames_list_cov_estimation_new
        """
        # Extend chunks that are too short
        if min_length_slice > 0:
            frames_list_cov_estimation_new = []
            for slice_frame in frames_list_cov_estimation:
                start_frame, end_frame = slice_frame
                if end_frame - start_frame + 1 < min_length_slice:
                    start_frame = max(0, start_frame - min_length_slice)
                frames_list_cov_estimation_new.append((start_frame, end_frame))
            return frames_list_cov_estimation_new

        return frames_list_cov_estimation

    @staticmethod
    def get_chunks_slices(L1_frames_all_chunks, dft_props=None, time_props=None, recursive_average=False):
        """
        Get the slices that tell you which frames of the signal to process in each chunk.

        If recursive_average is True, then all frames are processed sequentially.

        Otherwise, the signal is divided into chunks, whose length is approximately 0.5s,
        and the voiced speech is processed in each chunk.
        """

        if recursive_average:
            all_slices = [(bf_start, bf_stop) for bf_start, bf_stop in
                             zip(range(0, L1_frames_all_chunks), range(1, L1_frames_all_chunks + 1),)]
            slice_bf_list = all_slices
            slice_cov_est_list = all_slices
        else:
            slice_bf_list, slice_cov_est_list, num_chunks = Manager.get_chunks_slices_block_processing(
                L1_frames_all_chunks, dft_props, time_props)

        slice_bf_list = [slice(bf_start, bf_stop) for bf_start, bf_stop in slice_bf_list]
        slice_cov_est_list = [slice(bf_start, bf_stop) for bf_start, bf_stop in slice_cov_est_list]
        num_chunks = len(slice_cov_est_list)

        if num_chunks < 1:
            raise ValueError(f"Number of chunks is {num_chunks}, but it should be at least 1.")

        if len(slice_bf_list) != len(slice_cov_est_list):
            raise ValueError("The number of beamforming slices and covariance estimation slices must be the same.")

        return slice_bf_list, slice_cov_est_list, num_chunks

    @staticmethod
    def get_chunks_slices_block_processing(L1_frames_all_chunks, dft_props=None, time_props=None):
        """ Block processing: divide the signal into chunks, whose length is approximately 0.5s,
        and process the voiced speech in each chunk. """

        L2_frames_chunk = Manager.get_frames_single_chunk(time_props, dft_props, L1_frames_all_chunks)
        if L1_frames_all_chunks < L2_frames_chunk:
            raise ValueError(f"{L1_frames_all_chunks = } must be greater than {L2_frames_chunk = }.")
        overlap_frames = int(time_props['overlap_frame_cov_est_percentage'] * L2_frames_chunk)

        # Get the slices that tell you which frames of the signal to process in each chunk

        # Find the chunks of the signal where the voiced speech is present and form the covariance estimation frames
        min_length_slice = time_props.get('min_length_slice', 1)

        if not time_props['fixed_length_chunks']:
            raise NotImplementedError("Deprecated")
            # max_chunk_len_frames = 100000
            # if sig_type != 'sinusoidal':
            #     max_chunk_len_frames = int(
            #         time_props['max_chunk_len_seconds'] * dft_props['fs'] / dft_props['r_shift_samples'])
            #
            # slice_bf_list, voiced_flag_list = (
            #     Manager.find_chunks_from_f0_over_time(f0_over_time, max_frames_chunk=max_chunk_len_frames,
            #                                           max_difference_f0=15))
            # slice_cov_est_list = Manager.get_cov_estimation_frames(slice_bf_list, voiced_flag_list,
            #                                                        min_length_slice=min_length_slice,
            #                                                        overlap_frames=overlap_frames)
        else:
            # Fixed length chunks. This way, though, some of the last frames might not be processed, which is undesirable.
            slice_bf_list = [(bf_start, bf_stop) for bf_start, bf_stop in
                             zip(range(0, L1_frames_all_chunks, L2_frames_chunk),
                                 range(L2_frames_chunk, L1_frames_all_chunks + L2_frames_chunk, L2_frames_chunk))]
            assert len(slice_bf_list) > 0

            if slice_bf_list[-1][0] >= L1_frames_all_chunks:
                slice_bf_list.pop()
                warnings.warn("The last chunk starts after the signal ends.")

            if slice_bf_list[-1][1] > L1_frames_all_chunks:
                slice_bf_list[-1] = (slice_bf_list[-1][0], L1_frames_all_chunks + 1)

            slice_cov_est_list = Manager.get_cov_estimation_frames(slice_bf_list, voiced_flag_list=None,
                                                                   min_length_slice=min_length_slice,
                                                                   overlap_frames=overlap_frames)

        return slice_bf_list, slice_cov_est_list, len(slice_bf_list)

    @staticmethod
    def find_chunks_from_f0_over_time(f0_over_time, max_frames_chunk=50, max_difference_f0=15):
        """
        Find chunks of the signal that have approximately constant f0.
        Return a list of tuples (start_sample, end_sample) for each chunk.
        Consecutive nans are considered as the same chunk.
        If valid numbers are separated by nans, they are considered as different chunks, unless they are shorter
        than min_num_frames_chunk

        # Example f0_over_time:
        # array([nan, nan, nan, nan, nan, nan, nan, 114.5, 113.9, 112.6, 109.4, 104.4, 109.4, 107.5, 98., 93., 89.9, 90.9,
        #        100.3, nan, nan, nan, nan, nan,
        #        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 110.6, 110., 108.7, 108.1, 106.3, 102.6, 107.5, nan,
        #        117.2, 116.5, 107.5, 105.6, 103.2, 102.,
        #        101.5, 99.7, 96.3, 90.4, 89.3, 93., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
        #        nan, nan, nan, nan,
        #        nan, 95.2, 93., 92., 92., 92., 89.9, 86.8, 85.8, 84.8, 81., 79.1, 80.1, 78.7, 78.2, 71.7, 65.4, nan, nan,
        #        nan, nan, nan, nan, ])
        """

        def f0_difference_small(idx_1, idx_2, max_difference_f0_=max_difference_f0):
            if np.isnan(f0_over_time[idx_1]) or np.isnan(f0_over_time[idx_2]):
                return True
            return np.abs(f0_over_time[idx_1] - f0_over_time[idx_2]) < max_difference_f0_

        if np.all(np.isnan(f0_over_time)):
            raise ValueError("f0_over_time is all NaNs. Cannot find chunks.")

        # if np.isfinite(f0_over_time[0]):
        #     raise NotImplementedError("The first frame of f0_over_time must be nan. It is not possible to start with a valid f0.")

        # The first frame will certainly start and finish with nans, until the first valid f0 is found
        # start_frame_0 = 0
        # end_frame_0 = np.where(np.isfinite(f0_over_time))[0][0]
        # slice_frames_list = [(start_frame_0, end_frame_0)]
        # voiced_unvoiced_flag_list = [False]
        # idx_frame = end_frame_0

        voiced_unvoiced_flag_list = []
        slice_frames_list = []
        idx_frame = 0
        while idx_frame < len(f0_over_time):
            is_voiced_frame = np.isfinite(f0_over_time[idx_frame])
            voiced_unvoiced_flag_list.append(is_voiced_frame)
            is_same_type = np.isfinite if is_voiced_frame else np.isnan

            start_frame = idx_frame
            end_frame = start_frame
            while (end_frame < len(f0_over_time) and
                   is_same_type(f0_over_time[end_frame]) and
                   f0_difference_small(start_frame, end_frame)) and \
                    end_frame - start_frame < max_frames_chunk:
                end_frame += 1
            slice_frames_list.append((start_frame, end_frame))
            idx_frame = end_frame

        # Add some extra frames at the end to make sure that the last chunk covers the whole signal
        slice_frames_list[-1] = (slice_frames_list[-1][0], slice_frames_list[-1][1] + 5)

        # Check that the slices are contiguous
        for idx, (start_frame, end_frame) in enumerate(slice_frames_list[:-1]):
            if end_frame != slice_frames_list[idx + 1][0]:
                raise ValueError(f"Frames are not contiguous: {end_frame} != {slice_frames_list[idx + 1][0]}.")

        return slice_frames_list, voiced_unvoiced_flag_list

    @staticmethod
    def add_noise_to_signals(signals_dict, noise_var, which_signals=None):
        raise NotImplementedError("This method is not used anymore. ")

        """ Add noise to the signals in signals_dict. """

        if which_signals is None:
            which_signals = 'wet_rank1'

        for key, sig_dict in signals_dict.items():
            if key in which_signals:
                signals_dict[key]['time'], _ = Manager.add_noise_snr(signals_dict[key]['time'], snr_db=noise_var)
                signals_dict[key]['stft'], _ = Manager.add_noise_snr(signals_dict[key]['stft'], snr_db=noise_var)
                signals_dict[key]['stft_masked'] = signals_dict[key]['stft']

        for key, sig_dict in signals_dict.items():
            if key in which_signals:
                signals_dict[key]['stft_conj'] = signals_dict[key]['stft'].conj()

        for key, sig_dict in signals_dict.items():
            if not np.allclose(sig_dict['stft'], sig_dict['stft_conj'].conj()):
                print(f"Distorted signals: {which_signals}")
                raise ValueError(f"{key = } stft and stft_conj are not conjugate symmetric.")

        return signals_dict

    @staticmethod
    def check_non_empty_dict(signals_unproc):
        # Check that fields_to_check are not empty (all zeros)

        fields_to_check = ['stft', 'stft_conj', 'time']
        optional_dicts = ['noise_freq_est']
        # fields_to_check = ['stft', 'stft_masked', 'stft_conj', 'time']

        errors_ = []
        for key, sig_dict in signals_unproc.items():
            if key in optional_dicts:
                continue
            for key_inner, sig in sig_dict.items():
                if key_inner in fields_to_check:
                    if np.allclose(sig, 0):
                        # Collect all keys that are all zeros in a list
                        errors_.append((key, key_inner))

        if errors_:
            warnings.warn(f"All zeros signals: {errors_}")

    @staticmethod
    def calculate_missing_fields_signals_dict(signals, needs_masked=True):
        """ Add display_name, stft_masked, and stft_conj to signals. In-place operation. """

        for key in signals.keys():
            signals[key]['display_name'] = pl.get_display_name(key)
            if needs_masked and 'stft_masked' not in signals[key]:
                signals[key]['stft_masked'] = copy.deepcopy(signals[key]['stft'])
            if 'stft_conj' not in signals[key]:
                signals[key]['stft_conj'] = copy.deepcopy(signals[key]['stft'].conj())

        Manager.check_non_empty_dict(signals)

        return signals

    @staticmethod
    def get_frames_single_chunk(time_props, dft_props, L1_frames_all_chunks):
        """ Get the number of frames in a single chunk. """

        if time_props['fixed_length_chunks'] and time_props['chunk_len_seconds'] >= time_props['duration_approx_seconds']:
            # Force the number of frames in a single chunk to be equal to the number of frames in the whole signal
            # This is useful when the signal is processed in a single chunk
            return L1_frames_all_chunks

        chunk_len = int(time_props['chunk_len_seconds'] * dft_props['fs'])  # Number of samples in a single chunk
        L2_frames_chunk = -1  # Number of frames in a single chunk

        # If fixed length chunks are used, then the number of frames in a single chunk is fixed
        # and is equal to the number of frames that fit in the chunk length
        if time_props['fixed_length_chunks']:
            L2_frames_chunk = int(chunk_len // dft_props['r_shift_samples'])
            print(f"Chunk length: {chunk_len / dft_props['fs']:.2f} seconds. "
                  f"Number of frames per chunk: {L2_frames_chunk}")

        return L2_frames_chunk

    @staticmethod
    def allocate_beamformed_signals(L1_frames_all_chunks, K_nfft_real, beamforming_methods):
        """ Allocate space for beamformed signals in the frequency domain (bfd = beamformed data). """
        bfd_all_chunks_stft = {bf_name: np.zeros((K_nfft_real, L1_frames_all_chunks), dtype=np.complex128)
                               for bf_name in beamforming_methods}

        # Masked: to evaluate performance on HARMONIC BINS only, to highlight improvements of cyclic beamforming.
        bfd_all_chunks_stft_masked = copy.deepcopy(bfd_all_chunks_stft)

        return bfd_all_chunks_stft, bfd_all_chunks_stft_masked
