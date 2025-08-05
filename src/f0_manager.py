import copy
import warnings
from enum import Enum
import numpy as np
from typing import Dict, Optional

from . import harmonic_info as hi
from .coherence_manager import CoherenceManager
from .sin_generator import SinGenerator
from .modulator import Modulator
from .spectral_estimator import SpectralEstimator
from .f0_estimator import F0Estimator

from src.globs import rng


class F0ChangeAmount(Enum):
    no_change = 'none'
    small = 'small'  # within 1% of the previous value: same note/phoneme, just some modulation
    large = 'large'  # more than 1% of the previous value: different note/phoneme, we reset the estimate

    # Define = (copy) operation such that it is always copied by value and not by reference
    def __copy__(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        return copy.deepcopy(self)

    def to_number(self):
        if self == F0ChangeAmount.no_change:
            return 0
        elif self == F0ChangeAmount.small:
            return 1
        elif self == F0ChangeAmount.large:
            return 2
        else:
            raise ValueError(f"Unknown F0ChangeAmount: {self}")


class F0Tracker:
    """ Class to keep track of the fundamental frequency and its modulation frequency. """

    def __init__(self):
        # Each list contains a value per each slice (or chunk) of the signal

        # self.alpha_smooth = np.array([])  # smoothed fundamental frequency (for beamforming)
        self.alpha_smooth_list = []  # smoothed fundamental frequency (for beamforming)
        self.alpha_data_list = []  # estimated fundamental frequency from the data
        self.mod_amount_list = []  # amount of change in the modulation frequency

        # These two are only for debugging
        self.mod_amount_list_number = []  # amount of change in the modulation frequency as a number
        self.mod_amount_data_list = []  # amount of change in the modulation frequency based on the data


class F0Manager:
    inharmonicity_signs = rng.choice([-1, 1], 1000)

    def __init__(self, f0_est: F0Estimator = None):
        if f0_est is not None:
            self.f0_est = f0_est
        else:
            self.f0_est = F0Estimator()

    @classmethod
    def shift_frequencies_by_percentage(cls, freqs, inharmonicity_percentage, all_same_sign=False, fixed_amount=False):
        """
        Inharmonicity is the deviation of the harmonics from the ideal harmonic series.
        If not fixed_amount, inharmonicity_percentage gives the maximum of the uniform distribution.
        """

        if inharmonicity_percentage == 0:
            return freqs

        signs = np.ones(len(freqs)) if all_same_sign else cls.inharmonicity_signs[:len(freqs)]
        if fixed_amount:
            freqs = freqs + inharmonicity_percentage * freqs * signs
        else:
            # Freqs are shifted by an amount which varies between
            # [freq*(1-inharmonicity_percentage), freq*(1+inharmonicity_percentage)]
            freqs = freqs + inharmonicity_percentage * freqs * (2 * rng.random(len(freqs)) - 1) * signs

        return freqs

    @classmethod
    def calculate_negative_harmonics_f0(cls, f0_over_time_slice, P_max, f0_err_percent_=0):

        alpha_vec_hz = np.array([0])

        fundamental_freq_hz_ = np.nan
        if np.any(np.isfinite(f0_over_time_slice)):
            fundamental_freq_hz_ = np.nanmean(f0_over_time_slice)
            alpha_vec_hz = np.array([n * fundamental_freq_hz_ for n in range(100)])

            # It is better to correlate the signal with downshifted versions of itself
            alpha_vec_hz = -alpha_vec_hz

            alpha_vec_hz = np.unique(alpha_vec_hz)
            if alpha_vec_hz.size == 0:
                alpha_vec_hz = np.array([0])

        # Sort alpha_vec_hz using the absolute values in ascending order
        alpha_vec_hz = np.r_[sorted(alpha_vec_hz, key=lambda x: abs(x))][:P_max]

        # Artificially add error to the alphas used for estimating the modulated signals
        if f0_err_percent_ != 0 and np.all(np.isfinite(alpha_vec_hz)) and len(alpha_vec_hz) > 1:
            alpha_vec_hz = cls.shift_frequencies_by_percentage(alpha_vec_hz, f0_err_percent_ / 100,
                                                               all_same_sign=True,
                                                               fixed_amount=True)
            if alpha_vec_hz.size > 0:
                fundamental_freq_hz_ = np.abs(alpha_vec_hz[1])

        return alpha_vec_hz, fundamental_freq_hz_

    # Commented out for now as not tested thoroughly
    # @classmethod
    # def get_alpha_from_synthetic_signal_harmonics(cls, freqs_synthetic_signal, alpha_max_hz, P_max, f0_err_percent_=0):
    #
    #     alpha_vec_hz = np.r_[0, freqs_synthetic_signal.copy()]
    #
    #     # It is better to correlate the signal with downshifted versions of itself
    #     alpha_vec_hz = -alpha_vec_hz
    #
    #     alpha_vec_hz = np.unique(alpha_vec_hz)
    #     if alpha_vec_hz.size == 0:
    #         raise ValueError("No harmonics found in the synthetic signal. What's up?.")
    #
    #     # Sort alpha_vec_hz using the absolute values in ascending order
    #     alpha_vec_hz = np.r_[sorted(alpha_vec_hz, key=lambda x: abs(x))][:P_max]
    #
    #     # Artificially add error to the alphas used for estimating the modulated signals
    #     if f0_err_percent_ != 0 and np.all(np.isfinite(alpha_vec_hz)) and len(alpha_vec_hz) > 1:
    #         alpha_vec_hz = cls.shift_frequencies_by_percentage(alpha_vec_hz, f0_err_percent_ / 100, all_same_sign=True,
    #                                                            fixed_amount=False)
    #
    #     return alpha_vec_hz

    @classmethod
    def calculate_modulation_amount(cls, alpha_measured, alpha_measured_old, alpha_modulation, alpha_thresholds=None,
                                    mod_amount_list=None):
        """ Calculate the amount of change in the modulation frequency. """

        if alpha_modulation.size == 0:
            return F0ChangeAmount.small

        # Calculate the amount of change in the modulation frequency based on the measured modulation frequency.
        fund_freq_measured = np.abs(alpha_measured[1]) if len(alpha_measured) > 1 else 0
        fund_freq_measured_old = np.abs(alpha_measured_old[1]) if len(alpha_measured_old) > 1 else 0
        diff_percent_data = np.abs(fund_freq_measured - fund_freq_measured_old) / (fund_freq_measured_old + 1e-6)
        mod_amount_data = cls.get_mod_amount_from_diff_percentage(diff_percent_data, alpha_thresholds)

        # Calculate the amount of change in the modulation frequency based on the smoothed modulation frequency.
        fund_freq_smooth = np.abs(alpha_modulation[1]) if len(alpha_modulation) > 1 else 0
        diff_percent_smooth = np.abs(fund_freq_measured - fund_freq_smooth) / (fund_freq_smooth + 1e-6)
        mod_amount_smooth = cls.get_mod_amount_from_diff_percentage(diff_percent_smooth, alpha_thresholds)

        # If the modulation frequency has changed significantly, ignore the harmonic structure of the signal.
        # If it has changed slightly, re-estimate the modulated signals.
        # Use the frequency at which the signal was modulated last to decide whether to re-estimate the modulated signals.
        mod_amount = mod_amount_data
        if mod_amount_list is not None and len(mod_amount_list) > 0:
            if mod_amount_list[-1].to_number() <= 1:
                # previous frame was no change or small change: use the smoothed modulation frequency
                # to decide whether to re-estimate the modulated signals.
                # Note that after a large change, a small change is always set.
                mod_amount = mod_amount_smooth

        return mod_amount

    @staticmethod
    def get_mod_amount_from_diff_percentage(change_percentage, alpha_thresholds):
        """ Get the amount of change in the modulation frequency based on the percentage difference. """

        if change_percentage < alpha_thresholds['no_change']:
            # modulation frequency has not changed at all.
            return F0ChangeAmount.no_change
        elif change_percentage < alpha_thresholds['small_change']:
            # modulation frequency is changing slightly: re-estimate the modulated signals
            return F0ChangeAmount.small
        else:
            # modulation frequency is changing a lot: ignore the harmonic structure of signal.
            # Wait for modulation to stabilize before using cyclostationary properties.
            return F0ChangeAmount.large

    @staticmethod
    def find_harmonic_bins(harmonics_hz=np.array([]), freq_range_allowed=(0, 8000),
                           max_relative_dist_from_harmonic=1.6, all_freqs_hz=np.array([]),
                           use_negative_frequencies=False):
        """
        Find bins that are processed with cyclic beamformers. They correspond to the harmonics of the signal.
        max_relative_distance_from_harmonic: maximum relative distance from a harmonic to be considered a harmonic
        freq_range_allowed: range of frequencies that are allowed to be processed by the beamformer
        """

        num_bins = len(all_freqs_hz) if use_negative_frequencies else len(all_freqs_hz) // 2 + 1
        harmonic_sets = -np.ones((num_bins,), dtype=int)
        if np.all(all_freqs_hz > 0):
            raise ValueError("This function expects the list of frequencies to contain negative frequencies as well. "
                             "Use np.fft.fftfreq(nfft, 1 / fs) to get the list of frequencies. ")

        if not np.all(np.isfinite(harmonics_hz)) or np.all(harmonics_hz <= 0):
            return np.array([]), harmonic_sets, np.array([]), np.arange(num_bins)

        # Remove harmonics that are outside the allowed frequency range
        harmonics_hz = harmonics_hz[harmonics_hz >= freq_range_allowed[0]]
        harmonics_hz = harmonics_hz[harmonics_hz <= freq_range_allowed[1]]

        if harmonics_hz.size == 0:
            harmonic_bins = np.array([])
            harmonic_frequencies_quantized_delta_f = np.array([])
            non_harmonic_bins = np.arange(num_bins)
            return harmonic_bins, harmonic_sets, harmonic_frequencies_quantized_delta_f, non_harmonic_bins

        delta_f = all_freqs_hz[1] - all_freqs_hz[0]
        max_distance_from_harmonic_hz = max_relative_dist_from_harmonic * delta_f

        non_harmonic_bins = []
        freqs_hz_positive = abs(all_freqs_hz[:len(all_freqs_hz) // 2 + 1])

        for kk, kk_hz in enumerate(freqs_hz_positive):
            harmonic_sets[kk] = np.argmin(np.abs(harmonics_hz - kk_hz))  # harmonic bin, assigned to set 0, ..., # signal harmonics-1

            harmonic_center_hz = harmonics_hz[harmonic_sets[kk]]
            bin_far_from_harmonic = np.abs(kk_hz - harmonic_center_hz) > max_distance_from_harmonic_hz
            bin_freq_too_high_or_too_small = (kk_hz < freq_range_allowed[0] or kk_hz > freq_range_allowed[1])
            if bin_freq_too_high_or_too_small or bin_far_from_harmonic:
                non_harmonic_bins.append(kk)
                harmonic_sets[kk] = -1  # not a harmonic bin, assigned to default set -1

        # Now we want to add to this list the negative frequencies corresponding to non-harmonics. This should be done
        # by exploiting symmetry around 0 Hz, ie symmetric around len(freqs_hz) // 2 + 1.
        # We will add the negative frequencies corresponding to the positive ones that were skipped.
        if use_negative_frequencies:
            reflected_bins = len(all_freqs_hz) - non_harmonic_bins
            reflected_bins = reflected_bins[reflected_bins > len(freqs_hz_positive) - 1]
            non_harmonic_bins = np.unique(np.concatenate((non_harmonic_bins, reflected_bins)))
            non_harmonic_bins = non_harmonic_bins[non_harmonic_bins < len(all_freqs_hz)]

        # Find corresponding freqs in Hz
        non_harmonic_bins = np.unique(np.array(non_harmonic_bins))
        harmonic_bins = np.setdiff1d(np.arange(num_bins), non_harmonic_bins)
        harmonic_frequencies_quantized_delta_f = all_freqs_hz[harmonic_bins]

        return harmonic_bins, harmonic_sets, harmonic_frequencies_quantized_delta_f, non_harmonic_bins

    @classmethod
    def calculate_harmonic_sets(cls, harmonic_frequencies_hz=np.array([]), max_rel_dist_from_harmonic=1,
                                freq_range_cyclic=(0, 8000), harmonics_est_algo='nls', nfft=None, fs=None,
                                num_mod_sets=None):
        """
        Calculate harmonic bins and sets from the estimated fundamental frequency.
        If estimate of all harmonics is provided, the fundamental frequency is not used.
        """

        harmonic_bins, harmonic_sets, _, _ = (
            cls.find_harmonic_bins(harmonic_frequencies_hz,
                                   freq_range_allowed=freq_range_cyclic,
                                   max_relative_dist_from_harmonic=max_rel_dist_from_harmonic,
                                   all_freqs_hz=np.fft.fftfreq(nfft, 1./fs),
                                   ))

        # For non-pathologic cases (f0 > delta_f), np.allclose(harmonic_sets, harmonic_sets_2) is True
        # harmonic_sets_2, _ = cls.get_harmonic_sets_old(harmonic_bins, nfft // 2 + 1)

        if 'f0' in harmonics_est_algo and 'both_dir' not in harmonics_est_algo:
            # assign everything to set 0 (all bins use same modulation freq: there is just one harmonic set)
            harmonic_sets[harmonic_sets != -1] = 0

        elif 'both_dir' in harmonics_est_algo:
            # There can't be more "harmonic sets" than the number of harmonics in the signal!
            # E.g., if signal only has fundamental and first harmonic, num_harmonics_signal = 2
            # Therefore harmonic sets should range from 0 to 1.
            num_harmonics_in_signal = min(num_mod_sets, harmonic_frequencies_hz.size)
            harmonic_sets[harmonic_sets >= num_harmonics_in_signal] = -1

        elif 'all_harmonics' in harmonics_est_algo:
            # There can't be more "harmonic sets" than the number of harmonics in the signal!
            # E.g., if signal only has fundamental and first harmonic, num_harmonics_signal = 2
            # Therefore harmonic sets should range from 0 to 1.
            num_harmonics_in_signal = harmonic_frequencies_hz.size
            if np.any(harmonic_sets >= num_harmonics_in_signal):
                raise ValueError("Too many harmonic sets?")
            harmonic_sets[harmonic_sets >= num_harmonics_in_signal] = -1

        else:
            raise ValueError(f"Unknown harmonics estimation algorithm: {harmonics_est_algo = }")

        return harmonic_bins, harmonic_sets

    @staticmethod
    def smoothen_alpha_vec_hz(alpha_new, alpha_old, forgetting_factor):
        if np.isfinite(alpha_old).all():
            if np.allclose(alpha_old, 0):
                return alpha_new
            if alpha_new.size != alpha_old.size:
                return alpha_new
            alpha_new = (1. - forgetting_factor) * alpha_old + forgetting_factor * alpha_new
        else:
            raise ValueError("Old alpha is not finite. ")

        return alpha_new

    def smoothen_alpha_vec_outer(self, alpha_vec_hz, alpha_list, forgetting_factor):
        if not np.all(np.isclose(alpha_vec_hz, 0)):
            alpha_vec_old = alpha_list[-1] if len(alpha_list) > 0 else alpha_vec_hz
            alpha_vec_hz = F0Manager.smoothen_alpha_vec_hz(alpha_vec_hz, alpha_vec_old,
                                                           forgetting_factor=forgetting_factor)
        return alpha_vec_hz

    @staticmethod
    def decide_modulation_frequency_and_amount(alpha_v_data, mod_amount_data, alpha_list_smooth, idx_chunk,
                                               mod_amount_previous):
        """ Decide whether to change the modulation frequency and how much to change it. """
        dcopy = copy.deepcopy

        # Small change or first chunk: read new modulation frequency from the data
        if idx_chunk == 0 or mod_amount_data == F0ChangeAmount.small:
            mod_amount = F0ChangeAmount.small
            alpha_v_smooth = dcopy(alpha_v_data)

        # Fun. freq. is changing too fast: keep the same modulation frequency as before and signal a large change
        elif mod_amount_data == F0ChangeAmount.large:
            mod_amount = F0ChangeAmount.large
            alpha_v_smooth = alpha_list_smooth[-1]

        else:
            # No change: keep the same modulation frequency as before and signal no change
            mod_amount = F0ChangeAmount.no_change
            alpha_v_smooth = alpha_list_smooth[-1]

        # Override no_change after large to enforce small change
        # If we have large change and then no change, we replace no-change -> small-change to force modulation
        if mod_amount == F0ChangeAmount.no_change and mod_amount_previous == F0ChangeAmount.large:
            mod_amount = F0ChangeAmount.small
            alpha_v_smooth = dcopy(alpha_v_data)

        return alpha_v_smooth, mod_amount

    @staticmethod
    def get_cyclostationary_waveform(signals_unproc, cyclostationary_target):
        """ Get the cyclostationary waveform and its type. """

        sig_nature = 'noise'
        signal_waveform = signals_unproc['noise']['time'][0]
        if cyclostationary_target:
            sig_nature = 'target'
            signal_waveform = signals_unproc['wet_rank1']['time'][0]

        return signal_waveform, sig_nature

    def get_modulation_sets_from_f0_tracker(self, f0_slice, idx_chunk, cfg_cyc, f0_tracker: F0Tracker,
                                            shift_both_directions=False):
        """ Get the modulation frequency and the amount of change in the modulation frequency. """

        # For real signals, estimate the modulation frequency from the fundamental frequency.
        dcopy = copy.deepcopy
        alpha_data_list = f0_tracker.alpha_data_list
        alpha_smooth_list = f0_tracker.alpha_smooth_list
        mod_amount_list = f0_tracker.mod_amount_list

        alpha_data, f0_mean = self.calculate_negative_harmonics_f0(f0_slice, cfg_cyc['P_max'], cfg_cyc['f0_err_percent'])

        mod_amount_data = (self.calculate_modulation_amount(
            alpha_measured=alpha_data,
            alpha_measured_old=alpha_data_list[-1] if alpha_data_list else np.array([]),
            alpha_modulation=alpha_smooth_list[-1] if alpha_smooth_list else np.array([]),
            mod_amount_list=mod_amount_list,
            alpha_thresholds=cfg_cyc['alpha_thresholds']))

        alpha_smooth, mod_amount = (
            self.decide_modulation_frequency_and_amount(alpha_data, mod_amount_data, alpha_smooth_list,
                                                        idx_chunk,
                                                        mod_amount_list[-1] if mod_amount_list else F0ChangeAmount.small))

        alpha_smooth_list.append(dcopy(alpha_smooth))
        alpha_data_list.append(dcopy(alpha_data))
        mod_amount_list.append(mod_amount)
        f0_tracker.mod_amount_data_list.append(mod_amount_data)
        f0_tracker.mod_amount_list_number.append(mod_amount.to_number())

        if not shift_both_directions or np.allclose(alpha_smooth, 0):  # downward shifts only or no modulations
            mods_list = [alpha_smooth]
        else:  # upward and downward shifts
            mods_list = []
            harmonics_arr = np.array([f0_mean*n for n in range(1, int(np.ceil(cfg_cyc['freq_range_cyclic'][1] / f0_mean)))])
            for pp in range(1, harmonics_arr.size + 1):
                mod_iter = []
                for cc in range(1-pp, harmonics_arr.size - pp + 1):
                    mod_iter.append(-cc * f0_mean)
                mods_list.append(dcopy(np.asarray(mod_iter)))

            # Sort by magnitude (0 is always first)
            mods_list = [
                np.asarray(sorted((x for x in arr), key=abs))
                for arr in mods_list
            ]

            # Keep at most P_max elements in each array
            mods_list = [x[:cfg_cyc['P_max']] for x in mods_list]

        assert [alpha_vec[0] == 0 for alpha_vec in mods_list], \
            "First element of mod vector must be 0. Otherwise problems copying multiband to narrowband covariances. "

        return mod_amount, mods_list

    @classmethod
    def compute_pairwise_differences_between_freqs(cls, x: np.ndarray, P_max=1000, negative_shifts_only=False) -> list:
        """
        The input is the list of harmonic frequencies of the synthetic signal.
        Output is a list of arrays.
        Each array contains the modulation frequencies for a harmonic set (dft bin and neighbours).
        Example:
            First array: modulations for the first harmonic set (fundamental frequency),
            second array: modulations for the second harmonic set, etc.

        Numeric example: input [ 200.,  401.,  598.,  799.]
        Output:
           (array([[   0., -201., -398., -599.],
                   [   0., -197.,  201., -398.],
                   [   0.,  197., -201.,  398.],
                   [   0.,  201.,  398.,  599.]]),

        """
        x = np.asarray(x)
        if not (np.all(np.isfinite(x)) and len(x) >= 1):
            raise ValueError(f"The input should be a list of harmonic frequencies of the synthetic signal, not {x = }.")

        diff_matrix = x[:, None] - x[None, :]
        if negative_shifts_only:
            raise ValueError("Better to use f0_both_dir as this one is a mix between using all harmonics and using only positive shifts ")
            warnings.warn("Warning! Using negative shifts only, this will make performance worse.")
            diff_matrix[diff_matrix > 0] = np.inf

        # Sort each row (axis=1) based on absolute values. If two elements have same absolute value, put the positive shift first/
        # Negative values correspond to downshifts, while positive values correspond to upshifts.
        # The rationale is that lower harmonic components might have higher power.
        # Only keep smallest P_max elements of each list.
        diff_lists_idx = np.lexsort((diff_matrix < 0, np.abs(diff_matrix)), axis=1)[:, :P_max]
        diff_matrix = np.take_along_axis(diff_matrix, diff_lists_idx, axis=1)

        # Reorganize in lists and remove elements that are not finite. Each modulation vector can have different length.
        diff_list = []
        for alpha_vec in diff_matrix:
            diff_list.append(alpha_vec[np.isfinite(alpha_vec)])

        return diff_list

    @classmethod
    def get_harmonic_sets_old(cls, harmonic_bins, nfft_real):
        """
        Return an array of length nfft_real with each harmonic bin assigned a set.
        Consecutive bins in harmonic_bins belong to the same set.
        For example, bins [12,13,14] are set 0, bins [24,25,26] are set 1.

        This function first creates an array filled with −1−1 to mark bins not present in harmonic_bins. It then sorts
        the provided harmonic_bins and iterates through them.
        If two successive bins are consecutive, they are assigned the same set; otherwise, the set index is incremented.
        """
        harmonic_sets = -np.ones(nfft_real, dtype=int)
        harmonic_sets_compact = np.zeros_like(harmonic_bins)
        if len(harmonic_bins) == 0:
            return harmonic_sets, harmonic_sets_compact

        set_ = 0
        harmonic_sets[harmonic_bins[0]] = set_
        # harmonic_bins[1:] goes from 1st to last element
        # harmonic_bins[:-1] goes from 0th to 2nd last element
        for current, previous in zip(harmonic_bins[1:], harmonic_bins[:-1]):
            if current == previous + 1:
                harmonic_sets[current] = set_
            else:
                set_ += 1
                harmonic_sets[current] = set_

        for kk in range(len(harmonic_bins)):
            harmonic_sets_compact[kk] = harmonic_sets[harmonic_bins[kk]]

        return harmonic_sets, harmonic_sets_compact

    def collect_info_for_crb_calculation(self, indices, cfg, dg):
        """
        Collect information for calculating the CRB.

        For a single sin: SNR ~= crb_dict['sin_amp_squared_crb'] / crb_dict['white_noise_var_crb']
        """

        crb_dict = {}
        crb_dict['white_noise_var_crb'] = np.var(dg.white_noise_crb)

        # Here, the signal is assumed to be real.
        # For complex signals, the factor is alpha^2, where alpha is the amplitude of the complex exponential.
        # The variance of real sinusoid is instead 0.5 * alpha^2, if alpha is the amplitude of the real sinusoid.
        crb_dict['sin_amp_squared_crb'] = (np.mean(dg.sin_amps_crb[:, indices], axis=-1) ** 2)

        # Number of samples used for frequency estimation
        crb_dict['N'] = min(len(indices), cfg['harmonics_est']['nfft'])

        return crb_dict

    def estimate_f0_or_resonant_freqs(self, signals, cfg, dft_props,
                                      sin_generators: Optional[Dict[str, SinGenerator]] = None,
                                      do_plots=False):
        """
        Estimate the fundamental frequency (f0) or resonant frequencies of the signal.
        # _f0 options: signals are modulated based on multiples of fundamental frequency (alpha_vec_hz). Worse performance.
        #  # all_harmonics options: the signals are modulated by the differences among harmonic. Better performance.
        """

        freqs_est = np.array([0])
        f0_over_time = np.array([0])
        crb_dict = {}
        harm_est_cfg = cfg['harmonics_est']
        waveform_cyclo, sig_nature = self.get_cyclostationary_waveform(signals, cfg['cyclostationary_target'])

        if cfg['cyclic']['P_max'] == 1:
            signals.pop('noise_freq_est', None)
            return freqs_est, crb_dict, f0_over_time

        if 'all_harmonics' in harm_est_cfg['algo']:

            if harm_est_cfg['algo'] == 'oracle_all_harmonics':  # use oracle information
                if cfg[sig_nature]['sig_type'] == 'sample':
                    raise ValueError(f"{harm_est_cfg['algo'] = } but {cfg[sig_nature]['sig_type'] = } "
                                     f"({sig_nature = }). This combination is invalid.")
                freqs_est = copy.deepcopy(sin_generators[sig_nature].freqs_synthetic_signal)

            else:  # estimate harmonics frequencies using periodogram 'periodogram_all_harmonics'

                sig_name_harmonic_est = harm_est_cfg['source_signal_name']
                sig_harmonic_est = signals[sig_name_harmonic_est]['time'][0]
                freqs_est, indices = SpectralEstimator.estimate_harmonics_periodogram(
                    sig_harmonic_est, harm_est_cfg, dft_props['fs'], do_plots)
                # crb_dict = self.collect_info_for_crb_calculation(indices, cfg, dg)

        else:  # Only estimate f0, not all harmonics individually
            # Warning: f0_over_time does not take the inharmonicity into account!
            f0_over_time = self.f0_est.get_f0_over_time(waveform_cyclo, sig_nature, cfg, dft_props,
                                                        f0_hz=sin_generators[sig_nature].f0,
                                                        do_plots=do_plots)

        # TODO is commenting out this code correct?
        # if cfg['cyclic']['f0_err_percent'] != 0 and not np.allclose(freqs_est, 0):
        #     freqs_est = self.shift_frequencies_by_percentage(
        #         freqs_est, cfg['cyclic']['f0_err_percent'] / 100, all_same_sign=True,
        #         fixed_amount=True)

        signals.pop('noise_freq_est', None)

        return freqs_est, crb_dict, f0_over_time

    def compute_modulation_sets(self, harmonic_freqs_est, cfg_harmonics_est, cfg_cyclic,
                                idx_chunk, f0_over_time_slice_bf, f0_tracker):

        if 'all_harmonics' in cfg_harmonics_est['algo']:
            # All harmonics provided, use their differences to compute the modulation list
            # Modulation sets = alpha_mods_list
            # Skip frequency tracking for synthetic signals. They have fixed fundamental frequency.

            if cfg_cyclic['P_max'] == 1:
                return F0ChangeAmount.small, [np.array([0])]

            if harmonic_freqs_est.size < 1:
                raise ValueError(f"No harmonics found in {harmonic_freqs_est = }.")

            freq_range_cyclic = cfg_cyclic['freq_range_cyclic']
            valid_freqs = (freq_range_cyclic[0] <= harmonic_freqs_est) & (harmonic_freqs_est <= freq_range_cyclic[1])

            # Compute modulation sets based on differences between estimated harmonic frequencies
            alpha_mods_sets = self.compute_pairwise_differences_between_freqs(harmonic_freqs_est[valid_freqs],
                                                                              cfg_cyclic['P_max'])

            mod_amount = F0ChangeAmount.small if idx_chunk == 0 else F0ChangeAmount.no_change

        else:
            shift_both_directions = 'both_dir' in cfg_harmonics_est['algo']
            # Only f0 provided, use the f0 to compute the modulation list. Tracking system.
            mod_amount, alpha_mods_sets = self.get_modulation_sets_from_f0_tracker(f0_over_time_slice_bf,
                                                                                   idx_chunk, cfg_cyclic, f0_tracker,
                                                                                   shift_both_directions)

        return mod_amount, alpha_mods_sets

    def compute_harmonic_and_modulation_sets_distance_based(self, harmonic_freqs_est, cfg_harmonics_est, cfg_cyclic, dft_props,
                                                            idx_chunk, f0_over_time_slice_bf, f0_tracker):
        """ Compute harmonic and modulation sets based on the estimated harmonic frequencies."""

        # Modulations applied to the signal depend on the differences between frequencies in "harmonic_freqs_est"
        # or on integer multiples of the fundamental frequency f0.
        mod_amount, alpha_modulation_sets = self.compute_modulation_sets(harmonic_freqs_est, cfg_harmonics_est,
                                                                         cfg_cyclic,
                                                                         idx_chunk, f0_over_time_slice_bf, f0_tracker)

        # Harmonic bins are close to harmonic frequencies in "harmonic_freqs_est"
        # Harmonic sets are collections of neighbouring harmonic bins
        harmonic_bins, harmonic_sets = \
            (self.calculate_harmonic_sets(harmonic_freqs_est, cfg_cyclic['max_relative_dist_from_harmonic'],
                                          cfg_cyclic['freq_range_cyclic'], cfg_harmonics_est['algo'], dft_props['nfft'],
                                          dft_props['fs'], num_mod_sets=len(alpha_modulation_sets)))

        assert len(alpha_modulation_sets) >= max(harmonic_sets)

        # Collect everything in a HarmonicInfo instance
        harmonic_info = hi.HarmonicInfo(harmonic_bins, harmonic_sets, harmonic_freqs_est, alpha_modulation_sets)

        return harmonic_info, mod_amount

    def get_harmonics_as_multiples_of_f0(self, f0_over_time_this_chunk, freq_range_cyclic):
        """ Get the harmonic frequencies as multiples of the fundamental frequency f0."""

        if np.allclose(f0_over_time_this_chunk, 0):
            return np.array([0])

        assert f0_over_time_this_chunk.size > 0
        fund_freq_mask_estimation = f0_over_time_this_chunk[0]

        # harmonics calculated based on fundamental_freq_hz if they are not provided explicitly
        harmonic_freqs_est = np.arange(0, freq_range_cyclic[1], fund_freq_mask_estimation)[1:]
        harmonic_freqs_est = harmonic_freqs_est[harmonic_freqs_est >= freq_range_cyclic[0]]

        return harmonic_freqs_est
