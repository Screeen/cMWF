import copy
import warnings

import librosa
import numpy as np
import scipy
from pathlib import Path

from . import utils as u
from .manager import Manager
from .globs import rng
from . import sin_generator
from .config import ConfigManager


class DataGenerator:

    cfg_man = ConfigManager()

    def __init__(self, target_harmonic_corr=1.0, noise_harmonic_corr=1.0, mean_random_proc=0.):
        self.cached_target_samples = {}
        self.cached_rirs = {}

        self.sin_amps_crb = np.array([])  # for CRB only
        self.white_noise_crb = 0  # for CRB only
        self.noise_freq_est = np.array([])  # for freq. estimation only

        correlation_factors = {'target': target_harmonic_corr, 'noise': noise_harmonic_corr}
        self.sin_gen = {key: sin_generator.SinGenerator(val, mean_random_proc)
                        for key, val in correlation_factors.items() }

    @staticmethod
    def make_noise_reverberant(target_reverberant, noise_anechoic, impulse_response=np.array([])):

        num_mics, num_samples = target_reverberant.shape
        if impulse_response.size > 0:
            noise_reverberant = scipy.signal.convolve(noise_anechoic, impulse_response,
                                                      mode='full')[..., :noise_anechoic.shape[-1]]
        else:
            warnings.warn(f"impulse response for noise not provided. noise_reverberant = noise_anechoic!")
            noise_reverberant = copy.deepcopy(noise_anechoic)

        # make noise_reverberant num_samples long
        noise_reverberant = u.pad_last_dim(noise_reverberant, num_samples)[..., :num_samples]

        return noise_reverberant

    def generate_self_and_dir_noises(self, wet_target, snr_db_dir, snr_db_self, sample_path,
                                     dir_noise_ir=np.array([]), fs=16000, max_offset_seconds=None,
                                     N_num_samples=-1, sig_type='sample', f0_hz=0, num_harmonics=1000,
                                     inharmonicity_percentage=0., harmonic_correlation=None):
        """ Generate the self noise and the directional noise. """

        if N_num_samples > 0:
            wet_target_pad = u.pad_last_dim(wet_target, N_num_samples)
        else:
            wet_target_pad = wet_target

        # Generate anechoic noise
        use_fixed_amplitudes_sin = False
        noise_anechoic, sin_amps = self.generate_or_load_anechoic_signal(wet_target_pad.shape[-1], fs=fs,
                                                                         sample_path=sample_path, sig_type=sig_type,
                                                                         offset_seconds=max_offset_seconds, f0_hz=f0_hz,
                                                                         num_harmonics=num_harmonics,
                                                                         inharmonicity_percentage=inharmonicity_percentage,
                                                                         fixed_amplitudes_sin=use_fixed_amplitudes_sin,
                                                                         sin_gen=self.sin_gen['noise'])

        # copy the generated noise before convolution (for freq. estimation and CRB)
        self.noise_freq_est = copy.deepcopy(noise_anechoic)
        self.sin_amps_crb = sin_amps  # store the amplitudes for CRB

        # Directional noise
        noise_directional = DataGenerator.make_noise_reverberant(wet_target_pad, noise_anechoic, dir_noise_ir)
        noise_directional, snr_coeff = Manager.rescale_noise_to_snr(wet_target_pad, noise_directional, snr_db_dir)

        # Self noise (microphone noise)
        _, self_noise = Manager.add_noise_snr(clean=wet_target_pad, snr_db=snr_db_self, fs=fs)
        sum_noises = noise_directional + self_noise

        # Mix the signals
        noisy = wet_target_pad + sum_noises

        if N_num_samples > 0:
            noisy = noisy[..., :N_num_samples]
            sum_noises = sum_noises[..., :N_num_samples]

        # if snr_db_dir_noise == 0:
        #     assert np.allclose(np.mean(np.abs(sum_noises) ** 2), np.mean(np.abs(wet_target_pad) ** 2), atol=1e-2)

        return noisy, sum_noises

    def generate_or_load_anechoic_signal(self, duration_samples, fs, sample_path=None, sig_type='sample',
                                         offset_seconds=None, f0_hz=100, num_harmonics=20, selected_range_samples=None,
                                         inharmonicity_percentage=0., fixed_amplitudes_sin=False,
                                         sin_gen: sin_generator.SinGenerator = None, angle=None, sample_name=None,
                                         harmonic_correlation=None):

        sin_amps = np.array([])
        if not isinstance(f0_hz, list):
            f0_hz = [f0_hz, f0_hz]
        if (sig_type == 'sinusoidal' or sig_type == 'sinusoidal_varying_freq') and (
                not [0 < x < fs / 2 for x in f0_hz]):
            raise ValueError(f"{f0_hz = } must be in the range (0, {fs / 2}).")

        if sig_type == 'sample':
            anechoic, anechoic_fs = u.load_audio_file_random_offset(sample_path, fs=fs,
                                                                    duration_samples=duration_samples,
                                                                    offset_seconds=offset_seconds,
                                                                    smoothing_window=False)
        elif sig_type == 'sinusoidal':
            anechoic, sin_amps = sin_gen.generate_harmonics_process(f0_hz, duration_samples, fs=fs,
                                                                    num_harmonics=num_harmonics,
                                                                    inharmonicity_percentage=inharmonicity_percentage,
                                                                    fixed_amplitudes=fixed_amplitudes_sin)

        # elif sig_type == 'sinusoidal_varying_freq':
        #     anechoic = DataGenerator.generate_harmonic_frequency_varying_sinusoid(f0_hz, duration_samples, fs=fs,
        #                                                                           num_harmonics=num_harmonics,
        #                                                                           freq_variation_percentage=0.5,
        #                                                                           freq_variation_period=0.5)

        elif 'white' in sig_type:
            anechoic = rng.normal(0, 1, duration_samples)
            if 'lp' in sig_type:
                cutoff_freq = int(sig_type.split(sep='_')[-1]) / (fs / 2)
                b, a = scipy.signal.butter(8, cutoff_freq, btype='lowpass', analog=False, output='ba')
                anechoic = scipy.signal.lfilter(b, a, anechoic)
            elif 'hp' in sig_type:
                cutoff_freq = int(sig_type.split(sep='_')[-1]) / (fs / 2)
                b, a = scipy.signal.butter(8, cutoff_freq, btype='highpass', analog=False, output='ba')
                anechoic = scipy.signal.lfilter(b, a, anechoic)
        else:
            raise ValueError(f"Invalid target type: {sig_type}")

        if np.all(~np.isfinite(anechoic)):
            warnings.warn(f"Signal is all invalid {sample_path = }.")

        anechoic[~np.isfinite(anechoic)] = 0
        anechoic = DataGenerator.make_silent_start_and_end(anechoic, selected_range_samples, duration_samples)
        anechoic = u.smoothen_corners(anechoic)
        anechoic, coeff = u.normalize_variance(anechoic, 1)
        sin_amps = sin_amps * coeff
        anechoic = anechoic[np.newaxis, :]
        anechoic = u.pad_last_dim(anechoic, duration_samples)

        # To save file as wav
        # scaled = np.int16(anechoic / np.max(np.abs(anechoic)) * 32767)
        # scipy.io.wavfile.write('sinusoidal.wav', 16000, scaled)

        return anechoic, sin_amps

    @staticmethod
    def make_silent_start_and_end(sig, selected_range_samples, target_duration):

        # To avoid corner effects of STFT, select the central portion of the signal, smoothen it, pre- and post- pad
        if selected_range_samples is None:
            selected_range_samples = (0, target_duration)

        anechoic_central_portion = sig[selected_range_samples[0]:selected_range_samples[1]]
        anechoic_central_portion = u.smoothen_corners(anechoic_central_portion, alpha=1)
        sig = np.pad(anechoic_central_portion,
                     (selected_range_samples[0], target_duration - selected_range_samples[1]))
        return sig

    def get_rirs(self, speech_ir_path, dir_noise_ir_path, M, rir_max_len, fs=16000, real_rirs_flag=True):
        # Load the impulse responses

        if real_rirs_flag:  # Load the impulse responses
            if self.cached_rirs.get(speech_ir_path) is None:
                self.cached_rirs[speech_ir_path], speech_ir_fs = librosa.load(speech_ir_path, sr=fs, mono=False)
            speech_ir = self.cached_rirs[speech_ir_path]

            dir_noise_ir = np.array([[]])
            if dir_noise_ir_path is not None:
                if self.cached_rirs.get(dir_noise_ir_path) is None:
                    self.cached_rirs[dir_noise_ir_path], dir_noise_ir_fs = librosa.load(dir_noise_ir_path, sr=fs,
                                                                                        mono=False)
                dir_noise_ir = self.cached_rirs[dir_noise_ir_path]
        else:
            non_zero_samples_rir = rir_max_len // 10
            speech_ir = rng.standard_normal(size=(M, non_zero_samples_rir))
            dir_noise_ir = rng.standard_normal(size=(M, non_zero_samples_rir))
            speech_ir = u.pad_last_dim(speech_ir, rir_max_len)
            dir_noise_ir = u.pad_last_dim(dir_noise_ir, rir_max_len)

        # speech_ir = speech_ir[:M, :rir_max_len]
        # dir_noise_ir = dir_noise_ir[:M, :rir_max_len]
        # Try to load a sample every two sensors, but check if there are enough samples
        if (speech_ir.size > 0 and speech_ir.shape[0] < M) or (dir_noise_ir.size > 0 and dir_noise_ir.shape[0] < M):
            raise ValueError(f"Not enough channels in the impulse response file: {speech_ir.shape[0]} < {M} or "
                             f"{dir_noise_ir.shape[0]} < {M}.")
        # elif speech_ir.shape[0] > 2 * M:  # load every other microphone to simulate a larger array
        #     speech_ir = speech_ir[0:2 * M:2, :rir_max_len]  # is the same as speech_ir[::2][:M]
        #     dir_noise_ir = dir_noise_ir[0:2 * M:2, :rir_max_len]
        else:
            speech_ir = speech_ir[:M, :rir_max_len]
            dir_noise_ir = dir_noise_ir[:M, :rir_max_len]

        # normalization = np.sum(speech_ir[0] ** 2)
        # normalization = 1
        # speech_ir = speech_ir / normalization  # Normalize the impulse response to unit energy
        # dir_noise_ir = dir_noise_ir / normalization  # Normalize the impulse response to unit energy

        return speech_ir, dir_noise_ir

    def generate_noise_noisy(self, reverberant_mix, noise_params, noise_cov_est_len, stft):
        # reverberant_mix will go into noisy: don't use wet_rank1 or early!

        noise_params_copy = copy.deepcopy(noise_params)
        sig_type = noise_params['sig_type']

        if sig_type == 'white_lp' or sig_type == 'white_hp':
            noise_params_copy['sig_type'] = '_'.join([sig_type, str(noise_params['white_lp_freq'])])

        if sig_type != 'sample':
            noise_params_copy['sample_path'] = None
        else:
            noise_params_copy['sample_path'] = self.get_sample_path(noise_params_copy['sample_name'])
            print(f"File loaded: {noise_params_copy['sample_path']}")

        # Remove this parameters so that "generate_self_and_dir_noises" doesn't throw TypeError
        pop_keys = ['white_lp_freq', 'sample_name', 'noise_var_rtf', 'skip_convolution_rir']
        for key in pop_keys:
            if key in noise_params_copy:
                noise_params_copy.pop(key)

        signal_names = ['noisy', 'noise', 'noise_cov_est']
        t = 'time'
        f = 'stft'
        f_conj = 'stft_conj'
        r = {key: {} for key in signal_names}

        r['noisy'][t], r['noise'][t], r['noise_cov_est'][t] = (
            self.generate_noise_mix_and_noise_cov_est(reverberant_mix, noise_cov_est_len, noise_params_copy))

        # Compute STFT and conjugate for the noisy signal, noise, and noise_cov_est
        for key in r.keys():
            r[key][f] = stft(r[key][t])
            r[key][f_conj] = r[key][f].conj()

        if r['noise_cov_est'][t].shape[-1] / noise_params['fs'] < 1:
            warnings.warn(f"Duration of the noise signal for covariance estimation is less than 1 second. "
                          f"Consider increasing the duration.")

        return r

    @staticmethod
    def generate_target_signal(anechoic, speech_ir, nfft_atf, stft_obj, mic0_idx=0, max_len_ir=None):
        # Generate the target signals: wet, wet_rank1, early

        def stft(x):
            return stft_obj.stft(x)

        if max_len_ir is None:
            max_len_ir = nfft_atf

        # Convolving the anechoic signal with the impulse response to get the wet signal
        # This contains the FULL reverberation, not just early reflections.
        # Only used to produce noisy signal.
        wet = scipy.signal.convolve(anechoic, speech_ir, mode='full')[:, :anechoic.shape[-1] + max_len_ir - 1]
        wet_stft = stft(wet)
        target_duration_samples = wet.shape[-1]
        target_duration_frames = wet_stft.shape[-1]

        # To have all signals have the same length, pad anechoic to the same length as wet (after convolution!)
        anechoic = u.pad_last_dim(anechoic, target_duration_samples)

        # Convolving the anechoic signal with the initial part of the impulse response to get the wet_rank1 signal
        wet_rank1 = scipy.signal.convolve(anechoic, speech_ir[:, :max_len_ir], mode='full')[:, :target_duration_samples]
        wet_rank1_stft = stft(wet_rank1)[..., :target_duration_frames]

        target_signal_dict = {
            'wet': {'time': wet, 'stft': wet_stft, 'stft_conj': np.conj(wet_stft)},
            'wet_rank1': {'time': wet_rank1, 'stft': wet_rank1_stft, 'stft_conj': np.conj(wet_rank1_stft)},
        }

        return target_signal_dict

    def generate_noise_mix_and_noise_cov_est(self, reverberant_target, noise_cov_est_len, noise_params_copy):
        # Noise realization 1 used for the mix
        # Noise is generated with the same length as the signal

        # We use a new DataGenerator object to generate the noise, so we can use the same phases and amplitudes
        # for the noise realization used for the mix and the noise realization used for the covariance estimation.
        noisy, noise = (self.generate_self_and_dir_noises(wet_target=reverberant_target,
                                                          **noise_params_copy,
                                                          N_num_samples=reverberant_target.shape[-1]))

        # Noise realization 2 used for the covariance estimation
        _, noise_cov_est = (self.generate_self_and_dir_noises(wet_target=reverberant_target,
                                                              **noise_params_copy,
                                                              N_num_samples=noise_cov_est_len))

        assert reverberant_target.shape[-1] == noisy.shape[-1] == noise.shape[-1]
        assert noise_cov_est_len == noise_cov_est.shape[-1]

        return noisy, noise, noise_cov_est

    @staticmethod
    def get_sample_path(target_file_name=None):
        project_path = Path(__file__).resolve().parent.parent
        datasets_path = project_path.parent / 'datasets'

        if target_file_name is not None and target_file_name.lower() != 'none':
            target_path = datasets_path / 'audio' / target_file_name

            # if target_path is not a valid wav file, look into subdirectories recursively
            if not target_path.exists():
                target_path = list(datasets_path.glob('**/' + target_file_name))[0]
            elif target_path.is_dir():
                possible_paths = list(target_path.glob('**/*.wav'))
                if len(possible_paths) == 0:
                    raise ValueError(f"No wav files found in {target_path}.")

                is_iowa_notes = any(['.ff.' in str(p) for p in possible_paths])
                if is_iowa_notes:
                    possible_paths_filtered = possible_paths.copy()
                    not_allowed = ['0', '1', '4', '5', '6', '7']  # filter higher and lower notes
                    for p in possible_paths:
                        if any([f'{n}.' in str(p) for n in not_allowed]):
                            possible_paths_filtered.remove(p)
                    possible_paths = possible_paths_filtered

                return rng.choice(possible_paths)

            return target_path

        else:
            raise ValueError(f"Invalid target file name: {target_file_name}")

    def generate_anechoic_signal(self, fs, target_sett, SFT):
        """ Generate an anechoic signal from a sample or a sinusoidal signal. """

        # To avoid corner effects of STFT, select the central portion of the signal, smoothen it, and pad it.
        target_sett['selected_range_samples'] = (2 * SFT.lower_border_end[0],
                                                 SFT.upper_border_begin(target_sett['duration_samples'])[0])

        anechoic_path = self.get_sample_path(target_sett['sample_name']) if target_sett[
                                                                                'sig_type'] == 'sample' else None
        fixed_amplitudes_sin = False
        anechoic, _ = self.generate_or_load_anechoic_signal(fs=fs, sample_path=anechoic_path,
                                                            fixed_amplitudes_sin=fixed_amplitudes_sin,
                                                            sin_gen=self.sin_gen['target'], **target_sett)
        # if anechoic_path:
        #     print(f"Generated anechoic signal from path: {anechoic_path}")

        return anechoic, anechoic_path

    @staticmethod
    def get_target_duration(time_props, dft_props, recursive_estimation=False):
        """ Get the target duration of the signal. """

        if recursive_estimation:
            chunk_len_seconds = dft_props['nfft'] / dft_props['fs']
        else:
            chunk_len_seconds = float(time_props['chunk_len_seconds'])

        fs = int(dft_props['fs'])
        duration_approx_seconds = float(time_props['duration_approx_seconds'])
        chunk_len = int(chunk_len_seconds * fs)

        if chunk_len_seconds < duration_approx_seconds:  # good, chunk is shorter than total duration
            num_chunks_approx = int(max(1, duration_approx_seconds / chunk_len_seconds))
            target_duration = int(num_chunks_approx * chunk_len)
        else:
            target_duration = int(chunk_len)

        return target_duration

    def generate_signals(self, cfg, SFT_real, dft_props):

        if cfg['cyclostationary_target'] and cfg['cov_estimation']['recursive_average'] and (
                cfg['time']['duration_approx_seconds'] > cfg['cov_estimation']['noise_cov_est_len_seconds']):
            raise NotImplementedError(f"Duration of the signal for covariance estimation must be greater than the "
                                      f"approximate duration of the signal with current implementation.")
        fs = dft_props['fs']
        cfg['target']['duration_samples'] = self.get_target_duration(cfg['time'], dft_props,
                                                                     cfg['cov_estimation']['recursive_average'])
        target_ir_path, dir_noise_ir_path = self.cfg_man.get_impulse_response_paths(cfg['target']['angle'],
                                                                                    cfg['rir_specs'])
        print(f"{target_ir_path = }")

        if cfg['noise'].get('skip_convolution_rir', False):
            dir_noise_ir_path = None

        target_wave, target_anechoic_path = self.generate_anechoic_signal(dft_props['fs'], cfg['target'], SFT_real)

        # Load the impulse response, convolve the target_wave signal with it and transform everything to the frequency domain
        speech_ir, dir_noise_ir = self.get_rirs(target_ir_path, dir_noise_ir_path,
                                                cfg['M'], int(cfg['rir_specs']['rir_len_seconds'] * fs), fs=fs,
                                                real_rirs_flag=cfg['rir_specs']['use_real_rirs'])

        # Generate the target signals: wet and wet_rank1 from target_wave and speech_ir
        max_len_ir_atf = int(np.round(dft_props['nfft'] * 4 / 5))
        target_signal_dict = self.generate_target_signal(target_wave, speech_ir, dft_props['nfft'], SFT_real, mic0_idx=0,
                                                         max_len_ir=max_len_ir_atf)

        # Add noise to the signal. Two different realizations for the noise.
        noise_params = {'dir_noise_ir': dir_noise_ir, 'fs': fs}
        noise_params.update(cfg['noise'])
        noisy_noise_dict = self.generate_noise_noisy(reverberant_mix=target_signal_dict['wet']['time'],
                                                     noise_params=noise_params,
                                                     noise_cov_est_len=int(
                                                         cfg['cov_estimation']['noise_cov_est_len_seconds'] * fs),
                                                     stft=SFT_real.stft)

        if (cfg['harmonics_est']['algo'] == 'periodogram_all_harmonics' and
                cfg['harmonics_est']['source_signal_name'] == 'noise_freq_est'):
            # For frequency estimation, we use a separate realization that is not used for the mix or for the covariance estimation.
            # In this way, we can access the true amplitudes of the harmonics for calculating the CRB.
            # Also, we can control the SNR of the harmonics WRT a white noise component, without affecting the SNR of
            # the mix.
            # snr_db_relative (noise_freq_est only):
            # how much is sinusoidal signal stronger than white noise coming from same direction
            self.noise_freq_est, self.white_noise_crb = Manager.add_noise_snr(
                self.noise_freq_est, snr_db=cfg['harmonics_est']['snr_db_relative'], fs=fs)
            noisy_noise_dict['noise_freq_est'] = {'time': np.array([]), 'stft': np.array([]), 'stft_conj': np.array([])}
            noisy_noise_dict['noise_freq_est']['time'] = self.noise_freq_est

        signals_unproc_ = {**noisy_noise_dict, **target_signal_dict}
        signals_unproc_ = Manager.calculate_missing_fields_signals_dict(signals_unproc_,
                                                                        needs_masked=cfg[
                                                                            'use_masked_stft_for_evaluation'])

        return signals_unproc_, max_len_ir_atf, speech_ir

    @staticmethod
    def calculate_ground_truth_rtf(target_signal_dict):
        """ Calculate the ground truth RTF from the target signal. """

        M, K_nfft_real = target_signal_dict['stft'].shape[:2]
        target_rtf = np.zeros((K_nfft_real, M), dtype=complex, order='F')

        for kk in range(K_nfft_real):
            _, ev = np.linalg.eigh(target_signal_dict['stft'][:, kk] @
                                   target_signal_dict['stft_conj'][:, kk].T)
            target_rtf[kk] = ev[:, -1] / (ev[0, -1] + 1.e-6)

        return target_rtf

    @staticmethod
    def get_stft_objects(dft_props):

        win_name = dft_props['win_name']

        fs = dft_props['fs']
        freqs_hz = np.fft.fftfreq(dft_props['nfft'], 1 / dft_props['fs'])
        window = scipy.signal.windows.get_window(win_name, dft_props['nw'], fftbins=True)

        SFT = scipy.signal.ShortTimeFFT(hop=dft_props['r_shift_samples'], fs=fs, win=window, fft_mode='twosided',
                                        scale_to='magnitude')

        SFT_real = scipy.signal.ShortTimeFFT(hop=dft_props['r_shift_samples'], fs=fs, win=window, fft_mode='onesided',
                                             scale_to='magnitude')

        if not scipy.signal.check_COLA(window, dft_props['nfft'], dft_props['noverlap']):
            raise ValueError('The window does not satisfy the COLA condition')

        if not scipy.signal.check_NOLA(window, dft_props['nfft'], dft_props['noverlap']):
            raise ValueError('The window does not satisfy the NOLA condition')

        return SFT, SFT_real, freqs_hz
