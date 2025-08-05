import warnings
import librosa
import numpy as np

from fast_nls_src import single_pitch as single_pitch_nls
from .config import ConfigManager

from . import plotter as pl


class F0Estimator:

    def __init__(self):
        self.cached_estimated_f0 = {}

    def calculate_or_estimate_f0(self, x, dft_props, sig_type='sample', f0_hz=0, f0_bounds_hz=(20, 400),
                                 f0_estimation_algo='nls',
                                 is_speech_signal=False):
        """ Calculate or estimate fundamental frequency of the signal. """

        # Generation of real or synthetic speech
        duration_samples = x.shape[-1]

        if 'white' in sig_type:
            f0_range = (0, 0, 0)
            f0_over_time = np.zeros(duration_samples // dft_props['r_shift_samples'])

        elif 'sinusoidal' in sig_type:
            f0_range = (f0_hz - 5, f0_hz, f0_hz + 5)
            f0_over_time = np.ones(duration_samples // dft_props['r_shift_samples']) * f0_hz

        else:
            assert x.ndim == 1 or x.shape[0] == 1, "Only one channel is supported for real speech."
            cached = ('signal' in self.cached_estimated_f0 and self.cached_estimated_f0.get(
                'signal').shape == x.shape and
                      np.allclose(self.cached_estimated_f0.get('signal'), x))
            if not cached:
                f0_range, f0_over_time = (
                    F0Estimator.find_f0_from_recording(x, dft_props['fs'], dft_props['r_shift_samples'],
                                                       dft_props['nfft'], is_speech_signal=is_speech_signal,
                                                       f0_estimation_algo=f0_estimation_algo,
                                                       f0_bounds_hz=f0_bounds_hz))
                self.cached_estimated_f0['signal'] = x
                self.cached_estimated_f0['f0_range'] = f0_range
                self.cached_estimated_f0['f0_over_time'] = f0_over_time
            else:
                f0_range = self.cached_estimated_f0['f0_range']
                f0_over_time = self.cached_estimated_f0['f0_over_time']

        return f0_range, f0_over_time

    @staticmethod
    def find_f0_from_recording(s, fs, R_shift_samples, nfft, f0_estimation_algo='pyin', f0_bounds_hz=(20, 400),
                               is_speech_signal=False):
        # TODO: maybe post-processing should be done after resampling f0_over_time to match the original signal
        # So we can freely set NaN without making resampling fail

        # if nfft / fs < 512 / 48000:
        #     warnings.warn(f"{nfft = } / {fs = } is too small. f0 estimation will be inaccurate. ")

        nfft_f0 = 2048
        r_shift_samples_f0 = nfft_f0 // 4
        if nfft > nfft_f0:
            nfft_f0 = nfft
            r_shift_samples_f0 = R_shift_samples

        f0_over_time_high_res, voiced_flag, voiced_probs = F0Estimator.find_f0_from_recording_internal(
            s, fs, r_shift_samples_f0, nfft_f0, f0_estimation_algo, f0_bounds_hz)

        f0_over_time_high_res, f0_range = F0Estimator.post_process_f0(s, f0_over_time_high_res, voiced_flag,
                                                                      voiced_probs,
                                                                      r_shift_samples_f0, nfft_f0, is_speech_signal, fs)

        if nfft == nfft_f0:
            return f0_range, f0_over_time_high_res
        else:
            # Now we need to upsample f0_over_time to match the original signal (nfft < nfft_f0)
            f0_over_time = librosa.core.resample(f0_over_time_high_res,
                                                 orig_sr=fs / r_shift_samples_f0,
                                                 target_sr=fs / R_shift_samples,
                                                 res_type='linear')

        return f0_range, f0_over_time

    @staticmethod
    def find_f0_from_recording_internal(s, fs, R_shift_samples, nfft, f0_estimation_algo='pyin',
                                        f0_bounds_hz=(80, 400)):

        if f0_estimation_algo == 'pyin':
            try:
                f0_over_time, voiced_flag_original, voiced_probs = librosa.pyin(s, sr=fs,
                                                                                fmin=f0_bounds_hz[0],
                                                                                fmax=f0_bounds_hz[1],
                                                                                hop_length=R_shift_samples)

            except librosa.util.exceptions.ParameterError:
                warnings.warn(f"Librosa could not estimate f0. Using default value.")
                f0_over_time = np.ones(s.size // R_shift_samples) * np.nan
                voiced_flag_original = np.zeros(s.size // R_shift_samples, dtype=bool)
                voiced_probs = np.zeros(s.size // R_shift_samples)

        elif f0_estimation_algo == 'penn':

            import penn
            import torch
            # filter everything below lp_freq_hz using torchaudio
            hopsize = R_shift_samples / fs
            fmin = 40
            fmax = 260
            gpu = None
            batch_size = 2048
            checkpoint = None
            center = 'half-hop'
            interp_unvoiced_at = 0
            decoder = 'viterbi'
            # audio = torchaudio.functional.highpass_biquad(audio, sample_rate, cutoff_freq=lp_freq_hz)
            s_tensor = torch.tensor(s[None, :], dtype=torch.float32, device='cpu')
            f0_over_time, voiced_probs = penn.from_audio(s_tensor, fs, hopsize=hopsize,
                                                         fmin=f0_bounds_hz[0], fmax=f0_bounds_hz[1],
                                                         checkpoint=checkpoint,
                                                         batch_size=batch_size, center=center,
                                                         interp_unvoiced_at=interp_unvoiced_at,
                                                         gpu=gpu)
            f0_over_time = f0_over_time[0].numpy()
            voiced_probs = voiced_probs[0].numpy()
            voiced_flag_original = voiced_probs > 0

            min_periodicity = np.min(voiced_probs)
            unvoiced_threshold = min(min_periodicity * 1.05, 0.001)
            f0_over_time[voiced_probs < unvoiced_threshold] = np.nan

        elif f0_estimation_algo == 'nls_f0':
            voiced_flag_original = np.ones(s.size // R_shift_samples, dtype=bool)
            voiced_probs = np.ones(s.size // R_shift_samples)

            f0_over_time = F0Estimator.non_linear_ls_f0(s, fs, R_shift_samples, nfft,
                                                        maxNoHarmonics=20, f0Bounds_hz=f0_bounds_hz, minModelOrder=2,
                                                        eps_refinement=1e-5)
            if f0_over_time.size < voiced_flag_original.size:
                f0_over_time = np.pad(f0_over_time, (0, voiced_flag_original.size - f0_over_time.size), 'edge')

        elif f0_estimation_algo == 'nls_bayesian_f0':
            f0_over_time, voiced_probs = F0Estimator.non_linear_ls_f0_bayesian(s, fs, R_shift_samples, nfft,
                                                                               maxNoHarmonics=20,
                                                                               f0Bounds_hz=f0_bounds_hz,
                                                                               minModelOrder=1, eps_refinement=1e-2)
            num_frames_desired = s.size // R_shift_samples
            if f0_over_time.size < num_frames_desired:
                f0_over_time = np.pad(f0_over_time, (0, num_frames_desired - f0_over_time.size), 'constant')
                voiced_probs = np.pad(voiced_probs, (0, num_frames_desired - voiced_probs.size), 'constant')

            voiced_flag_original = voiced_probs > 0.5
        else:
            raise ValueError(f"Unknown f0 estimation algorithm: {f0_estimation_algo}")

        return f0_over_time, voiced_flag_original, voiced_probs

    @staticmethod
    def post_process_f0(s, f0_over_time, voiced_flag_original, voiced_probs, R_shift_samples, nfft,
                        is_speech_signal=False,
                        fs=None):
        """ Post-process f0 estimation by: 1) setting f0 to NaN if S_pow is too small, 2) setting NaN values to mean. """

        value_invalid = np.nan  # if set to 0, it alters mean of f0 which is important for batch processing

        # Assume beginning and end of the signal are not voiced
        num_ignored_frames = min(1, f0_over_time.size // 4)
        voiced_probs[:num_ignored_frames] = 0
        voiced_probs[-num_ignored_frames:] = 0

        S_pow_stft = np.abs(librosa.stft(s, n_fft=nfft, hop_length=R_shift_samples)) ** 2

        high_bin = nfft
        # Consonants can be treated as silence. Consonants have more high-frequency content than vowels.
        if is_speech_signal:  # only consider up to 3Khz
            high_bin = int(1000 / fs * nfft)

        S_pow_stft = S_pow_stft[:high_bin, :f0_over_time.shape[-1]]

        # Set f0 to NaN if S_pow is too small
        S_pow_threshold = np.min(S_pow_stft) + 1e-4 * (np.max(S_pow_stft) - np.min(S_pow_stft))

        # average power over frequency bins to find quiet frames
        f0_over_time[np.mean(S_pow_stft, axis=0) < S_pow_threshold] = value_invalid

        f0_delta = np.nanstd(f0_over_time) / 2
        finite_f0 = np.isfinite(f0_over_time)

        if np.allclose(voiced_flag_original, False) or np.isclose(np.sum(voiced_probs[finite_f0]), 0):
            warnings.warn("No voiced frames found?")
            f0_range = (0, 0, 0)
        else:
            f0_mean = np.average(f0_over_time[finite_f0], weights=voiced_probs[finite_f0])
            f0_min, f0_max = f0_mean - f0_delta, f0_mean + f0_delta
            if np.abs(f0_max - f0_min) > 100:
                warnings.warn(f"Estimated f0 range is too large: {f0_min:.2f} - {f0_max:.2f}")
            f0_range = (f0_min, f0_mean, f0_max)

        # Assign a valid number to NaN so that we can resample!
        # If we set the mean, the mean of the f0 is not altered when doing batch processing.
        # However, the mean is unknown in an online setting, thus setting to 0 is a more realistic approach.
        # f0_over_time[~np.isfinite(f0_over_time)] = f0_range[1]
        f0_over_time[~np.isfinite(f0_over_time)] = 0

        return f0_over_time, f0_range

    @staticmethod
    def non_linear_ls_f0(speechSignal, samplingFreq, R_shift_samples, segmentLength, maxNoHarmonics=20,
                         f0Bounds_hz=(80, 400), minModelOrder=4, eps_refinement=1e-5):

        if samplingFreq == 16000 and segmentLength < 1024:
            warnings.warn(f"Segment length {segmentLength} is too short for 16kHz sampling frequency. "
                          f"Consider using longer segments for f0 estimation. ")

        if speechSignal.ndim > 1:
            speechSignal = speechSignal[:, 0]  # take only one channel
        nData = speechSignal.shape[0]

        # Consider overlapping segments
        nSegments = int(np.floor(nData / R_shift_samples))
        f0Bounds = np.array(f0Bounds_hz) / samplingFreq  # normalized frequency (1.0 = Nyquist frequency)

        f0Estimator = single_pitch_nls.single_pitch(segmentLength, maxNoHarmonics, f0Bounds)

        # do the analysis with overlapping segments
        start_indices = np.arange(nSegments) * R_shift_samples
        end_indices = start_indices + segmentLength
        f0Estimates = np.zeros((nSegments,))
        modelOrders = np.zeros((nSegments,))
        for ii in range(nSegments):
            speechSegment = np.array(speechSignal[start_indices[ii]:end_indices[ii]], dtype=np.float64)
            f0Estimates[ii] = (samplingFreq / (2 * np.pi)) * f0Estimator.est(speechSegment,
                                                                             lnBFZeroOrder=1, eps=eps_refinement)
            modelOrders[ii] = f0Estimator.modelOrder()

        f0Estimates[f0Estimates < f0Bounds_hz[0]] = np.nan
        f0Estimates[f0Estimates > f0Bounds_hz[-1]] = np.nan
        f0Estimates[modelOrders < minModelOrder] = np.nan
        modelOrders[modelOrders < minModelOrder] = np.nan

        return f0Estimates

    @staticmethod
    def non_linear_ls_f0_bayesian(speechSignal, samplingFreq, R_shift_samples, segmentLength, maxNoHarmonics=20,
                                  f0Bounds_hz=(80, 400), minModelOrder=4, eps_refinement=1e-5):

        if speechSignal.ndim > 1:
            speechSignal = speechSignal[:, 0]  # take only one channel
        nData = speechSignal.shape[0]

        # Consider overlapping segments
        nSegments = int(np.floor(nData / R_shift_samples))
        f0Bounds = np.array(f0Bounds_hz) / samplingFreq  # normalized frequency (1.0 = Nyquist frequency)

        f0Estimator = single_pitch_bayesian.single_pitch(segmentLength, maxNoHarmonics, f0Bounds,
                                                         trans_std_var=0.5,
                                                         voicing_prob=0.7)

        # do the analysis with overlapping segments
        start_indices = np.arange(nSegments) * R_shift_samples
        end_indices = start_indices + segmentLength
        f0Estimates = np.zeros((nSegments,))
        modelOrders = np.zeros((nSegments,))
        voicing_prob = np.zeros((nSegments,))

        #

        for ii in range(nSegments):
            speechSegment = np.array(speechSignal[start_indices[ii]:end_indices[ii]], dtype=np.float64)
            [f0Estimates[ii], modelOrders[ii], voicing_prob[ii]] = f0Estimator.est(speechSegment,
                                                                                   lnBFZeroOrder=0, eps=eps_refinement)

        f0Estimates = f0Estimates * (samplingFreq / (2 * np.pi))
        f0Estimates[f0Estimates < f0Bounds_hz[0]] = np.nan
        f0Estimates[f0Estimates > f0Bounds_hz[-1]] = np.nan
        f0Estimates[modelOrders < minModelOrder] = np.nan
        modelOrders[modelOrders < minModelOrder] = np.nan

        f0Estimates[voicing_prob < 0.7] = np.nan
        modelOrders[voicing_prob < 0.7] = np.nan

        return f0Estimates, voicing_prob

    def get_f0_over_time(self, signal_waveform, target_or_noise: str, cfg, dft_props, f0_hz=None, do_plots=False):
        """ Calculate fundamental frequency of the target or the noise signal. For the WHOLE signal. """

        signal_cfg = cfg[target_or_noise]

        if ConfigManager.is_speech(signal_cfg):
            is_speech_signal = True
            f0_bounds_hz = (80, 400)
        else:
            is_speech_signal = False
            f0_bounds_hz = (60, 500)

        if f0_hz is not None:
            if signal_cfg['sig_type'] == 'sinusoidal':
                if 'oracle' not in cfg['harmonics_est']['algo']:
                    warnings.warn("Non-oracle method specified, but using oracle f0 nonetheless.")
                return np.ones(signal_waveform.shape[-1] // dft_props['r_shift_samples']) * f0_hz
            else:
                raise ValueError("f0_hz should only be set for sinusoidal signals. ")

        _, f0_over_time = self.calculate_or_estimate_f0(signal_waveform,
                                                        dft_props,
                                                        sig_type=signal_cfg['sig_type'],
                                                        f0_hz=f0_hz,
                                                        f0_bounds_hz=f0_bounds_hz,
                                                        f0_estimation_algo=cfg['harmonics_est']['algo'],
                                                        is_speech_signal=is_speech_signal)

        if ConfigManager.get_plot_settings(cfg['plot'])['f0_spectrogram'] and do_plots:
            pl.plot_f0_spectrogram_outer(signal_waveform, f0_over_time, dft_props,
                                         signal_cfg['sample_name'][:-4])

        return f0_over_time
