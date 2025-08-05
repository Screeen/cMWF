import numpy as np

from . import utils as u
from . import f0_manager
from .globs import rng


class SinGenerator:
    def __init__(self, correlation_factor=1., mean_random_process=0):
        # Generate sequence of phases and amplitudes to be used when generating processes with same statistics
        max_num_harmonics = 1000

        self.mean_random_process = mean_random_process

        self.phases = rng.uniform(-np.pi, np.pi, max_num_harmonics)
        self.scaling_factors = rng.uniform(1, 10, max_num_harmonics)
        self.f0 = None
        self.freqs_synthetic_signal = np.array([])
        self.correlation_factor = correlation_factor

    def get_f0_synthetic_signal(self, f0_range, fs=16000):
        if self.f0 is None:
            self.f0 = rng.uniform(f0_range[0], f0_range[1])
            if not [0 < x < fs / 2 for x in f0_range]:
                raise ValueError(f"{f0_range = } must be in the range (0, {fs / 2}).")
        return self.f0

    def generate_harmonics_process(self, f0_range, num_samples, fs=16000, num_harmonics=20,
                                   inharmonicity_percentage=0., fixed_amplitudes=False):

        assert num_samples > 0, "num_samples must be positive."
        assert num_harmonics > 0, "num_harmonics must be positive."

        freqs_hz_sin = self.get_frequencies_harmonic_process(f0_range, num_harmonics, inharmonicity_percentage, fs)

        # It makes a big difference if PHASES are deterministic or random.
        # If they are random, sinusoids in noise_for_mix will have a phase relationship,
        # while sinusoids in noise_for_cov_est will have a different phase relationship.
        # This mismatch leads to a higher beamforming error, especially for cMVDR and cLCMV.
        # If they are deterministic, the phase relationship is the same for both noise_for_mix and noise_for_cov_est.
        # This gives better performance for all estimators.
        # phases = np.linspace(-np.pi / 4, np.pi, len(freqs_hz_sin), endpoint=False) + rng.uniform(-np.pi, np.pi)

        phases = self.phases[:len(freqs_hz_sin)]  # + rng.uniform(-np.pi, np.pi)
        scaling_factors = self.scaling_factors[:len(freqs_hz_sin)]
        x, amps = u.generate_harmonic_process(freqs_hz_sin, num_samples, fs=fs, phases=phases,
                                              corr_factor=self.correlation_factor,
                                              scaling_factors=scaling_factors, fixed_amplitudes=fixed_amplitudes,
                                              mean_harmonic=self.mean_random_process,
                                              local_rng=rng)

        return x, amps

    def get_frequencies_harmonic_process(self, f0_range, num_harmonics, inharmonicity_percentage, fs):

        f0 = self.get_f0_synthetic_signal(f0_range, fs)
        assert f0 > 0, "f0 must be positive."

        if self.freqs_synthetic_signal.size == 0:
            freqs_hz_sin = np.array([f0 * n for n in range(1, num_harmonics + 1)])
            if inharmonicity_percentage > 0:
                freqs_hz_sin = f0_manager.F0Manager.shift_frequencies_by_percentage(freqs_hz_sin,
                                                                                    inharmonicity_percentage / 100.,
                                                                                    all_same_sign=False,
                                                                                    fixed_amount=True)
                freqs_hz_sin[0] = f0  # keep the fundamental frequency the same
            freqs_hz_sin = freqs_hz_sin[freqs_hz_sin < fs / 2 - 100]  # avoid aliasing
            self.freqs_synthetic_signal = freqs_hz_sin.copy()

        return self.freqs_synthetic_signal

    # @staticmethod
    # def generate_harmonic_frequency_varying_sinusoid(f0, num_samples, fs=16000, num_harmonics=20,
    #                                                  freq_variation_percentage=0.05, freq_variation_period=0.5):
    #     assert f0 > 0, "f0 must be positive."
    #     assert num_samples > 0, "num_samples must be positive."
    #     assert num_harmonics > 0, "num_harmonics must be positive."
    #
    #     # Use code above to generate a frequency-varying sinusoid
    #     freqs_hz_sin = np.array([f0 * n for n in range(1, num_harmonics + 1)])
    #     freqs_hz_sin = freqs_hz_sin[freqs_hz_sin < fs / 2 - 100]
    #
    #     time_axis = np.linspace(0, num_samples / fs, num_samples, endpoint=False)
    #     frequency_variation = freq_variation_percentage * f0 * np.sin(
    #         2 * np.pi * time_axis / freq_variation_period)
    #
    #     freqs_hz_sin = freqs_hz_sin[:, np.newaxis] + frequency_variation
    #
    #     # phases = np.linspace(-np.pi / 4, np.pi, freqs_hz_sin.shape[0], endpoint=False)
    #     phases = rng.uniform(-np.pi, np.pi, freqs_hz_sin.shape[0])
    #     x = u.generate_harmonic_process(freqs_hz_sin, num_samples, fs=fs, phases=phases, correlated_amplitudes=True)
    #
    #     return x
