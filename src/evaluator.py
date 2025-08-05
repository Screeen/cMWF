import copy
import warnings
from pathlib import Path

import numpy as np
import pesq as pypesq  # pip install https://github.com/ludlows/python-pesq/archive/master.zip
import pystoi
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import pysepm_evo

from . import utils as u
from . import plotter as pl
from .manager import Manager

metrics_need_db = ['rmse', 'mae', 'mmse', 'sinr']

# List of signals that should not be evaluated
names_signals_to_skip = ['wet', 'noise', 'early', 'noise_cov_est', 'wet_rank1']


class Evaluator:

    def __init__(self, reference, processed_dict, metrics, fs=16000):
        self.reference = reference
        self.processed_dict = processed_dict
        self.fs = fs
        self.metrics = metrics

        # Check that dimensions are correct
        for sig_name, signal in self.processed_dict.items():
            if signal.shape != self.reference.shape:
                warnings.warn(
                    f"Signal shape {signal.shape} is different from reference shape {self.reference.shape} for "
                    f" {sig_name} signal.")

    def evaluate(self):
        # Time domain metrics
        # metrics = ['stoi', 'pesq', 'rmse', 'mae', 'mmse', 'fwsnrseg', 'snrseg']
        if len(self.metrics) == 0:
            return {}

        r = {eval_method: {beamformer_name: -np.inf for beamformer_name in self.processed_dict.keys()}
             for eval_method in self.metrics}
        for sig_name, signal in self.processed_dict.items():

            if signal.shape != self.reference.shape:
                raise ValueError(
                    f"Signal shape {signal.shape} is different from reference shape {self.reference.shape} for "
                    f" {sig_name} signal.")

            if 'stoi' in self.metrics:
                stoi = -np.inf
                if np.sum(np.abs(signal)) > 0:
                    stoi = pystoi.stoi(x=self.reference, y=signal, fs_sig=self.fs, extended=True)
                r['stoi'][sig_name] = stoi

            if 'pesq' in self.metrics:
                pesq_res = 0
                try:
                    if np.sum(np.abs(signal)) > 0:
                        if self.fs == 16000:
                            pesq_res = pypesq.pesq(ref=self.reference, deg=signal, fs=self.fs)
                        elif self.fs == 8000:
                            pesq_res = pypesq.pesq(ref=self.reference, deg=signal, fs=self.fs, mode='nb')
                except pypesq.cypesq.NoUtterancesError as e:
                    warnings.warn(f"Error computing PESQ: {e}")
                r['pesq'][sig_name] = pesq_res

            if any(['mos' in x for x in self.metrics]):
                dnsmos_res = -np.inf
                if np.sum(np.abs(signal)) > 0:
                    try:
                        signal = u.normalize_volume(signal, 0.95)
                        dnsmos_res = dnsmos.run(signal, sr=self.fs)
                    except Exception as e:
                        warnings.warn(f"Error computing DNSMOS: {e}")
                r['ovrl_mos'][sig_name] = dnsmos_res['ovrl_mos']
                r['sig_mos'][sig_name] = dnsmos_res['sig_mos']
                r['bak_mos'][sig_name] = dnsmos_res['bak_mos']

            if 'rmse' in self.metrics:
                # rms_accuracy = -np.inf
                rms_error = 0
                if np.sum(np.abs(signal)) > 0:
                    diff = np.abs(self.reference - signal) ** 2
                    # rms_error = np.sqrt(np.mean(diff[diff > 0]))  # doing this makes all errors equally bigger
                    rms_error = np.sqrt(np.mean(diff))
                    # rms_accuracy = -rms_error
                # r['rmse'][sig_name] = 10 * np.log10(rms_error + 1e-8)
                r['rmse'][sig_name] = rms_error

            if 'mae' in self.metrics:  # Mean Absolute Error
                mae = -np.inf
                mae = 0
                if np.sum(np.abs(signal)) > 0:
                    mae = np.mean(np.abs(self.reference - signal))
                # r['mae'][sig_name] = 10 * np.log10(mae + 1e-8)
                r['mae'][sig_name] = mae

            if 'mmse' in self.metrics:
                # This metric expects signals in STFT domain
                # if signal.ndim < 2:
                #     raise ValueError("Why is the signal 1D? It should be 2D for calculating MMSE.")
                # K_nfft, L_num_frames = signal.shape
                # error_arr = np.ones(K_nfft) * np.inf
                # for kk in range(K_nfft):
                #     error_arr[kk] = np.mean(np.abs(self.reference[kk] - signal[kk]) ** 2)

                error_arr = np.abs(self.reference - signal) ** 2

                if error_arr.size > 0:
                    error_arr = np.mean(error_arr)
                    # r['mmse'][sig_name] = 10 * np.log10(error_arr + 1e-8)
                    r['mmse'][sig_name] = error_arr
                else:
                    r['mmse'][sig_name] = np.nan

            if 'fwsnrseg' in self.metrics:
                fwsnrseg = -np.inf
                if np.sum(np.abs(signal)) > 0:
                    fwsnrseg = pysepm_evo.fwSNRseg(self.reference, signal, self.fs)
                r['fwsnrseg'][sig_name] = fwsnrseg

            if 'sisdr' in self.metrics:
                sisdr = -np.inf
                if np.sum(np.abs(signal)) > 0:
                    sisdr = self.si_sdr(self.reference, signal)
                r['sisdr'][sig_name] = sisdr

            if 'snrseg' in self.metrics:
                snrseg = -np.inf
                if np.sum(np.abs(signal)) > 0:
                    snrseg_all = pysepm_evo.SNRseg(self.reference, signal, self.fs)
                    snrseg = np.mean(snrseg_all)
                r['snrseg'][sig_name] = snrseg

        # for eval_method, results in r.items():
        # Check for NaNs
        # for sig_name, result in results.items():
        #     if np.any(np.isnan(result)):
        #         warnings.warn(f"NaN value found in {eval_method} for {sig_name}.")

        return r

    @staticmethod
    def si_sdr(reference, estimation):
        """
        Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

        Args:
            reference: numpy.ndarray, [..., T]
            estimation: numpy.ndarray, [..., T]

        Returns:
            SI-SDR

        [1] SDR– Half- Baked or Well Done?
        http://www.merl.com/publications/docs/TR2019-013.pdf

        # >>> np.random.seed(0)
        # >>> reference = np.random.randn(100)
        # >>> si_sdr(reference, reference)
        # inf
        # >>> si_sdr(reference, reference * 2)
        # inf
        # >>> si_sdr(reference, np.flip(reference))
        # -25.127672346460717
        # >>> si_sdr(reference, reference + np.flip(reference))
        # 0.481070445785553
        # >>> si_sdr(reference, reference + 0.5)
        # 6.3704606032577304
        # >>> si_sdr(reference, reference * 2 + 1)
        # 6.3704606032577304
        # >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
        # nan
        # >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
        # array([6.3704606, 6.3704606])

        """
        estimation, reference = np.broadcast_arrays(estimation, reference)
        reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

        # This is $\alpha$ after Equation (3) in [1].
        optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

        # This is $e_{\text{target}}$ in Equation (4) in [1].
        projection = optimal_scaling * reference

        # This is $e_{\text{res}}$ in Equation (4) in [1].
        noise = estimation - projection

        ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
        return to_db(ratio, min_val=1.e-3)

    @staticmethod
    def print_results_from_dict(results_dict):
        """ Print a table where each row is an algorithm to evaluate and each column is a metric """
        if not results_dict:
            return

        table = PrettyTable()
        metric_names = list(results_dict.keys())
        algo_names = list(results_dict[metric_names[0]].keys())
        table.add_column("Algorithms", algo_names, align='l')
        for metric_name in metric_names:

            metric_name_no_postfix = metric_name
            if '_freq' or '_time' in metric_name:
                metric_name_no_postfix = metric_name.split('_')[0]

            col_data = []
            best_val = 1e12  # min(results_dict[metric_name].values())
            for algo_name in algo_names:
                val = results_dict[metric_name][algo_name]
                if metric_name_no_postfix.lower() in metrics_need_db:
                    val = to_db(val)
                col_data.append(f"{val:.3f}*" if np.abs(val - best_val) < 0.01 else f"{val:.3f} ")
            metric_name_print = f"{metric_name} [dB]" if metric_name.lower() in metrics_need_db else metric_name
            table.add_column(f"{metric_name_print}", col_data)
        table.float_format = '.3'
        print(table)
        print("------------------------------------------------------------------------------------------")

    # @staticmethod
    # def compute_snrs(weights, ch, max_signal_power, max_ratio_silent_frames=1e-4):
    #     """ Compute the SNR for each beamformer in weights. """
    #
    #     K = ch.cov_noisy_nb.shape[0]
    #     M = ch.cov_noisy_nb.shape[-1]
    #     P = ch.cov_noisy_wb.shape[-1] // M
    #     eps = 1e-8
    #
    #     weights_padded = {}
    #     for bf_name, w in weights.items():
    #         bf_display_name = h.get_display_name(
    #             bf_name)  # Convert the internal name of the algorithm to the display name
    #         MP_current = w.shape[0]
    #         weights_padded[bf_display_name] = w if MP_current == M * P else np.pad(w, ((0, M * P - MP_current), (0, 0)))
    #
    #     sig_pows = {bf_name: np.zeros(K) for bf_name in weights_padded.keys()}
    #     noise_pows = {bf_name: np.zeros(K) for bf_name in weights_padded.keys()}
    #     snrs_singleband = {bf_name: np.zeros(K) for bf_name in weights_padded.keys()}
    #     snrs_singleband_db = {bf_name: np.zeros(K // 2 + 1) for bf_name in weights_padded.keys()}
    #     snrs_fullband = {bf_name: 0 for bf_name in weights_padded.keys()}
    #
    #     for bf_name, w_pad in weights_padded.items():
    #         for kk in range(K):
    #             w = w_pad[:, kk]
    #             is_loud_bin = ch.cov_wet_wb[kk, 0, 0] > max_ratio_silent_frames * max_signal_power
    #             if is_loud_bin:
    #                 sig_pow = np.real(w.conj().T @ ch.cov_wet_wb[kk] @ w)
    #                 noise_pow = np.real(w.conj().T @ ch.cov_noisy_wb[kk] @ w + eps)
    #                 sig_pows[bf_name][kk] = sig_pow
    #                 noise_pows[bf_name][kk] = noise_pow
    #                 snrs_singleband[bf_name][kk] = sig_pow / noise_pow
    #                 if kk < K // 2 + 1:
    #                     snrs_singleband_db[bf_name][kk] = 10 * np.log10(snrs_singleband[bf_name][kk])
    #         snrs_fullband[bf_name] = np.nansum(sig_pows[bf_name]) / (np.nansum(noise_pows[bf_name]) + eps)
    #
    #     # snrs_avg = {bf_name: np.nansum(snrs_singleband[bf_name]) for bf_name in snrs_singleband.keys()}
    #     # snrs_fullband_db = {bf_name: 10 * np.log10(snrs_fullband[bf_name] + eps) for bf_name in snrs_fullband.keys()}
    #
    #     # if P > 1:
    #     #     Evaluator.plot_dict(snrs_db)
    #     #     Evaluator.plot_dict(snrs)
    #
    #     return snrs_fullband

    @staticmethod
    def plot_dict(x_dict):

        lines_names = list(x_dict.keys())
        values = np.array(list(x_dict.values()))

        fig, ax = plt.subplots()
        for i, line in enumerate(values):
            ax.plot(line, label=lines_names[i], linewidth=0.5)
        ax.legend()
        ax.grid()
        ax.set_ylabel('SNR [dB]')

        plt.show()

    @staticmethod
    def compute_differences_wrt_baseline(results_by_metric, algorithms, unprocessed_signal_name='Noisy'):
        """
        Compute the difference between the results of each algorithm and the unprocessed signal.
        :param results_by_metric:
        :return:
        """

        if unprocessed_signal_name not in algorithms:
            raise ValueError(f"Baseline algorithm {unprocessed_signal_name} not found in {algorithms = }")

        idx_unprocessed_signal = algorithms.index(unprocessed_signal_name)

        results_difference = {}
        for metric_name, results_array in results_by_metric.items():
            results_difference[metric_name] = np.zeros_like(results_array)
            for algo_idx, algo_results in enumerate(results_array):
                results_difference[metric_name][algo_idx] = algo_results - results_array[idx_unprocessed_signal]

        # Delete the baseline algorithm from the results
        for metric_name, results_array in results_difference.items():
            results_difference[metric_name] = np.delete(results_array, idx_unprocessed_signal, axis=0)
        algorithms_no_baseline = algorithms.copy()
        algorithms_no_baseline.remove(unprocessed_signal_name)

        return results_difference, algorithms_no_baseline

    @classmethod
    def rearrange_and_average_results(cls, results_dict, varying_param_values, metrics_list, baseline_name=None):
        """ Results are transferred from a dict to a numpy array and averaged over the realizations. """

        results_by_metric, algorithms = prepare_array_plots(results_dict, varying_param_values, metrics_list)
        is_relative_result = False

        if baseline_name:
            is_relative_result = True
            results_by_metric, algorithms = cls.compute_differences_wrt_baseline(
                results_by_metric, algorithms, unprocessed_signal_name=baseline_name)

        results_by_metric_diff_avg, _ = average_results(results_by_metric)

        return results_by_metric_diff_avg, algorithms, is_relative_result

    # @classmethod
    # def compute_sinr_single_beamformer(cls, cov_holder: CovarianceHolder, w_in, is_narrowband=True):
    #
    #     # Compute the SINR for a single beamformer w.
    #     K = cov_holder.cov_noisy_nb.shape[0]
    #     M = cov_holder.cov_noisy_nb.shape[-1]
    #     sinr = np.zeros(K, dtype=float)
    #     sinr_db = np.zeros(K, dtype=float)
    #     sig_pow = np.zeros(K, dtype=float)
    #     noise_pow = np.zeros(K, dtype=float)
    #
    #     w = w_in
    #     target_cov = cov_holder.cov_wet_nb
    #     noise_cov = cov_holder.cov_noise_wb_eval[:, :M, :M]
    #     if not is_narrowband:
    #         # w = w_in[:M]
    #         target_cov = cov_holder.cov_wet_wb
    #         noise_cov = cov_holder.cov_noise_wb_eval
    #
    #     for kk in range(K):
    #         # if cov_holder.cov_wet_nb[kk].shape[-1] != M:
    #         #     raise ValueError(f"Dimension mismatch: {cov_holder.cov_wet_nb[kk].shape[-1]} != {M} for kk={kk}")
    #         target_power = np.real(w[:, kk].conj().T @ target_cov[kk] @ w[:, kk])
    #         noise_power = np.real(w[:, kk].conj().T @ noise_cov[kk] @ w[:, kk])
    #         sinr[kk] = target_power / noise_power
    #         sinr_db[kk] = to_db(target_power) - to_db(noise_power)
    #         sig_pow[kk] = target_power
    #         noise_pow[kk] = noise_power
    #
    #     return sinr
    @classmethod
    def print_final_results_table(cls, results_dict, algorithms_no_baseline, varying_param_values):

        # Create and print the table
        table = PrettyTable()
        table.field_names = ["Algorithms"] + [str(x) for x in varying_param_values]

        for row_idx, row_name in enumerate(algorithms_no_baseline):
            row_data = [results_dict[row_idx, col_idx][0] for col_idx in range(len(varying_param_values))]
            table.add_row([row_name] + row_data)

        table.float_format = ".3"
        table.align = "r"

        print(table)

    @classmethod
    def get_metrics_list(cls, results_dict, varying_param_values):
        """ Get the list of metrics to evaluate. """

        metrics_list = []
        if len(varying_param_values) > 0 and len(results_dict) > 0:
            first_param_values = str(varying_param_values[0])
            if results_dict[first_param_values] and results_dict[first_param_values][0]:
                metrics_list = list(results_dict[first_param_values][0].keys())

        return metrics_list

    @staticmethod
    def mask_signals_for_eval(signals, harm_info, slice_bf, K_nfft_real):
        """ Add 'stft_masked' field to the signals_ground_truth_dict.
        Cannot be moved to evaluation unless we store the mask_harmonic_bins. """

        for key in signals.keys():
            signals[key]['stft_masked'][..., :K_nfft_real, slice_bf] = (
                    signals[key]['stft'][..., :K_nfft_real, slice_bf] *
                    harm_info.mask_harmonic_bins[:, None])

        return signals


# @staticmethod
def average_results(res_by_metric):
    """
    Compute the mean and standard deviation of the error over the realizations.
    Also compute the standard error of the mean (https://en.wikipedia.org/wiki/Standard_error.)
    and the 95% confidence interval.
    The confidence interval can be expressed in terms of probability with respect to a single theoretical
    (yet to be realized) sample: "There is a 95% probability that the 95% confidence interval calculated from a
    given future sample will cover the true value of the population parameter."
    """

    num_algos, num_varying_param_values, num_montecarlo = res_by_metric[next(iter(res_by_metric))].shape

    # This dictionary will contain the average results for each metric.
    res_by_metric_avg = {key: np.zeros((num_algos, num_varying_param_values, 3)) for key, value in
                         res_by_metric.items()}

    # This dictionary will contain the average results for each metric, for each realization.
    res_by_metric_avg_single_realizations = {key: np.zeros((num_algos, 3)) for key, value in
                                             res_by_metric.items()}

    axes_avg_list = [(-1), (-1, -2)]

    for metric in res_by_metric.keys():

        for axes_avg in axes_avg_list:

            num_realizations = num_montecarlo
            dest = res_by_metric_avg[metric]
            if isinstance(axes_avg, tuple):
                num_realizations = num_montecarlo * num_varying_param_values
                dest = res_by_metric_avg_single_realizations[metric]

            stderr = 0
            if num_realizations > 1:
                stderr = np.nanstd(res_by_metric[metric], ddof=1, axis=axes_avg) / np.sqrt(num_realizations)
            avg = np.mean(res_by_metric[metric], axis=axes_avg)

            # 95% confidence interval
            avg_min_std = avg - 1.96 * stderr
            avg_plus_std = avg + 1.96 * stderr
            # avg_min_std[avg_min_std < 0] = avg

            dest[..., 0] = avg
            dest[..., 1] = avg_min_std
            dest[..., 2] = avg_plus_std

    return res_by_metric_avg, res_by_metric_avg_single_realizations


def to_db(x, min_val=1e-15):
    if np.any(x < 0):
        raise ValueError("Input to to_db should be non-negative.")
    return (10. * np.log10(x + min_val) + 300) - 300


def convert_results_to_db(results_all_metrics, metrics_list):
    metrics_to_convert = copy.deepcopy(metrics_need_db)
    metrics_to_convert = [f"{metric}_{domain}" for metric in metrics_to_convert for domain in ['time', 'freq']]
    metrics_to_convert += ['freq-err-mae']

    # Convert the results to dB
    results_db = copy.deepcopy(results_all_metrics)
    for metric in metrics_list:
        if metric.lower() in metrics_to_convert:
            # Remember that dest[..., 0] = avg, dest[..., 1] = avg_min_std, dest[..., 2] = avg_plus_std
            # Dimension 1 is avg - std_dev. If it goes under 0, replace it with just avg.
            dest = results_all_metrics[metric]
            if np.any(dest[..., 1] <= 0):
                dest[..., 1][dest[..., 1] <= 0] = dest[..., 0][dest[..., 1] <= 0]
            try:
                results_db[metric] = to_db(results_all_metrics[metric])
            except ValueError as e:
                warnings.warn(f"Error converting {metric} to dB: {e}: {results_all_metrics[metric]} for {metric = }")
                results_db[metric] = to_db(np.abs(results_all_metrics[metric]))

    return results_db


def prepare_array_plots(results_dict, varying_param_values, metrics_list):
    """
    Convert results_dict to a new dictionary where
    - the keys are the metrics
    - the values are numpy arrays of shape (num_algorithms, num_varying_param_values, num_montecarlo).

    :param results_dict:
    :param varying_param_values:
    :param metrics_list:
    :return: result_by_metric, algorithms
    result_by_metric: dictionary of numpy arrays with shape (num_algorithms, num_varying_param_values, num_montecarlo)
    algorithms: list of algorithms, in the same order as the first dimension of the numpy arrays.
    """

    first_mtc = 0
    num_montecarlo = min(len(results_dict[str(param)]) for param in varying_param_values)
    first_variation = next(iter(results_dict))
    first_metric = next(iter(results_dict[first_variation][first_mtc]))
    algorithms = list(results_dict[first_variation][first_mtc][first_metric])

    result_by_metric = {metric: np.zeros((len(algorithms), len(varying_param_values), num_montecarlo))
                        for metric in metrics_list}

    # Fill the numpy arrays with the data.
    for param_idx, param_val in enumerate(varying_param_values):
        for mtc_idx in range(num_montecarlo):
            for metric in metrics_list:
                for algo_index, algo in enumerate(algorithms):
                    param_val_str = str(param_val)
                    result_by_metric[metric][algo_index, param_idx, mtc_idx] = results_dict[param_val_str][mtc_idx][metric][algo]

    return result_by_metric, algorithms


def bake_dict_for_evaluation(signals_dict, mic0_idx=0, needs_masked_stft=True):
    """ Prepare signals_dict for evaluation. """
    # 1) Check that the signals_dict is not empty
    # 2) Normalize and pad the time domain signals
    # 3) Select only the first microphone for evaluation purposes

    Manager.check_non_empty_dict(signals_dict)

    # Select only the first microphone for evaluation purposes
    for sig_name in signals_dict.keys():
        if 'stft' in signals_dict[sig_name] and signals_dict[sig_name]['stft'].ndim == 3:
            signals_dict[sig_name]['stft'] = signals_dict[sig_name]['stft'][mic0_idx]
            if needs_masked_stft:
                signals_dict[sig_name]['stft_masked'] = signals_dict[sig_name]['stft_masked'][mic0_idx]
        if 'time' in signals_dict[sig_name] and signals_dict[sig_name]['time'].ndim == 2:
            signals_dict[sig_name]['time'] = signals_dict[sig_name]['time'][mic0_idx]

    # Check that frequency-domain signals correspond to real time-domain signals.
    # To do so, verify is STFT domain signals are mirrored around the center frequency.
    # for key in signals_dict.keys():
    #     is_symmetric = u.corresponds_to_real_time_domain_signal(signals_dict[key]['stft'][np.newaxis])
    #     if not is_symmetric:
    #         warnings.warn(f"Frequency domain signal {key} does not correspond to a real time domain signal.")

    # Remove from the dict all the algorithms for which algorithms_dict[algorithm] is False (inactive algorithms).
    # If key is not present *at all* in algorithms_dict, then the algorithm is not removed. This is useful for the noise signal.
    # for bf_name in bf_names_selected:
    #     if bf_name in signals_dict:
    #         signals_dict.pop(bf_name)

    # Normalize and pad the time domain signals
    new_len = max([value['time'].shape[-1] for value in signals_dict.values()])
    for key in signals_dict.keys():
        # signals_dict[key]['time'] = u.normalize_and_pad(signals_dict[key]['time'], new_len)
        signals_dict[key]['time'] = u.pad_last_dim(signals_dict[key]['time'], new_len)

    return signals_dict


# def evaluate_sinr_single_chunk(cov_holder: CovarianceHolder, weights: dict, use_masked_stft_for_evaluation=False, chunk_idx=0,
#                                sinr_dict: dict = None, harmonic_bins=np.array([])):
#     # Compute the SINR for each beamformer in weights.
#     # The SINR is computed based on the covariance matrices in cov_holder.
#     # Skip the evaluation of all signals that are in the skip_evaluation_list.
#     # if use_masked_stft_for_evaluation: raise a warning that the evaluation is not possible.
#
#     bf_names = list(weights.keys()) + ['noisy']
#     sinr_dict_before_avg = {bf_name: np.zeros(cov_holder.cov_noisy_nb.shape[0]) for bf_name in bf_names}
#
#     for bf_name, w in weights.items():
#         is_narrowband = 'cmwf' not in bf_name and 'cmvdr' not in bf_name
#         if bf_name in names_signals_to_skip:
#             continue
#
#         # bf_display_name = h.get_display_name(bf_name)
#         sinr = Evaluator.compute_sinr_single_beamformer(cov_holder, w, is_narrowband=is_narrowband)
#         sinr_dict_before_avg[bf_name] = sinr
#
#     M, K = cov_holder.cov_noisy_nb.shape[-1], cov_holder.cov_noisy_nb.shape[0]
#     w_noisy = np.zeros((M, K))
#     w_noisy[0, :] = 1
#     sinr_dict_before_avg['noisy'] = Evaluator.compute_sinr_single_beamformer(cov_holder, w_noisy, is_narrowband=True)
#
#     # u.plot([sinr_dict_before_avg['cmwf_oracle'], sinr_dict_before_avg['cmwf_semi-oracle']], titles=['Oracle', 'Semi-oracle'], subplot_height=1.5)
#     # u.plot([sinr_dict_before_avg['cmwf_oracle'][harmonic_bins], sinr_dict_before_avg['cmwf_semi-oracle'][harmonic_bins]], titles=['Oracle', 'Semi-oracle'], subplot_height=1.5)
#
#     for bf_name in sinr_dict_before_avg.keys():
#         if harmonic_bins.size > 0 and use_masked_stft_for_evaluation:
#             sinr_dict[bf_name][chunk_idx] = np.mean(sinr_dict_before_avg[bf_name][harmonic_bins])
#         else:
#             sinr_dict[bf_name][chunk_idx] = np.mean(sinr_dict_before_avg[bf_name])
#
#     return sinr_dict


def adjust_metrics_list(metrics_list_time_, is_speech=True):
    """ If signal is not speech, remove STOI and PESQ from the list of metrics to evaluate. """

    metrics_list_time = copy.deepcopy(metrics_list_time_)
    if not is_speech:
        remove_metrics = ['stoi', 'pesq', 'fwsnrseg']
        metrics_list_time = [x for x in metrics_list_time_ if x not in remove_metrics]

    return metrics_list_time


def evaluate_signals(signals_dict, metrics_list_time, metrics_list_freq, fs, use_masked_stft_for_evaluation=False,
                     K_nfft_real=None, reference_sig_name='wet_rank1', print_results=False):
    """ Evaluate all signals in signals_dict. """

    # Skip the evaluation of all signals that are in the skip_evaluation_list
    # Keep only the time signals for evaluation.
    # Plus, {value['display_name']: value['time'] } means that the dictionary will be {display_name: time_signal}
    signals_time_evaluation_dict = {value['display_name']: value['time'] for key, value in signals_dict.items() if
                                    key not in names_signals_to_skip}

    # Evaluate the signals in the time domain
    ev_time = Evaluator(reference=signals_dict[reference_sig_name]['time'],
                        processed_dict=signals_time_evaluation_dict,
                        metrics=metrics_list_time, fs=fs)
    res_time = ev_time.evaluate()

    # Evaluate the signals in the frequency domain

    # Choose whether to use the masked or unmasked STFT for evaluation.a
    # Masked evaluation emphasizes the performance on the harmonic components.
    stft_or_stft_masked = 'stft_masked' if use_masked_stft_for_evaluation else 'stft'

    signals_freq_evaluation_dict = {value['display_name']: value[stft_or_stft_masked][:K_nfft_real]
                                    for key, value in signals_dict.items() if key not in names_signals_to_skip}

    ev_freq = Evaluator(reference=signals_dict[reference_sig_name][stft_or_stft_masked][:K_nfft_real],
                        processed_dict=signals_freq_evaluation_dict, metrics=metrics_list_freq, fs=-1)
    res_freq = ev_freq.evaluate()

    combined_res = combine_results(res_time, res_freq, print_results)

    return combined_res


def combine_results(res_time, res_freq, print_results=False):
    res_combined_temp = {}
    for key in res_time.keys():
        key_new = '_'.join([key, 'time'])
        res_combined_temp[key_new] = res_time[key]

    for key in res_freq.keys():
        key_new = '_'.join([key, 'freq'])
        res_combined_temp[key_new] = res_freq[key]

    # Merge the results from the two evaluators to a single dictionary
    res_combined = {**res_combined_temp}

    if print_results:
        Evaluator.print_results_from_dict(res_combined)

    return res_combined


def rearrange_and_average_results_all(results_dict, plot_args_default, results_freq_est=None, baseline_name='noisy'):

    # Evaluate the beamformed signals
    varying_param_values = plot_args_default['varying_param_values']
    metrics_list = Evaluator.get_metrics_list(results_dict, varying_param_values)
    plot_args_bf = {}
    if metrics_list:
        results_by_metric_diff_avg, algorithms_no_baseline, is_relative_result = (
            Evaluator.rearrange_and_average_results(results_dict, varying_param_values, metrics_list, baseline_name=baseline_name))

        plot_args_bf = copy.deepcopy(plot_args_default)
        plot_args_bf.update({
            'result_by_metric': results_by_metric_diff_avg,
            'result_by_metric_db': convert_results_to_db(results_by_metric_diff_avg, metrics_list),
            'metrics_list': metrics_list,
            'algorithms': algorithms_no_baseline,
            'is_relative_result': is_relative_result,
        })

    # Evaluate the frequency estimation
    plot_args_freq_est = {}
    if results_freq_est:
        metrics_list_freq = Evaluator.get_metrics_list(results_freq_est, varying_param_values)
        if metrics_list_freq:
            results_freq_est_diff, algorithms_no_baseline_freq_est, is_relative_result_freq = (
                Evaluator.rearrange_and_average_results(results_freq_est, varying_param_values, metrics_list_freq))

            plot_args_freq_est = copy.deepcopy(plot_args_default)
            plot_args_freq_est.update({
                'result_by_metric': results_freq_est_diff,
                'result_by_metric_db': convert_results_to_db(results_freq_est_diff, metrics_list_freq),
                'metrics_list': metrics_list_freq,
                'algorithms': algorithms_no_baseline_freq_est,
                'is_relative_result': is_relative_result_freq,
            })

    return plot_args_bf, plot_args_freq_est


def evaluate_frequency_estimation(ref_freqs, est_freqs):
    """ Evaluate the frequency estimates """

    # Count how many frequencies are missing in the estimated frequencies
    num_misses = max(0, ref_freqs.size - est_freqs.size)

    # Compute absolute differences between all pairs in ref_sort and est_sort
    diff_matrix = np.abs(ref_freqs[:, np.newaxis] - est_freqs[np.newaxis, :])

    # diff_matrix = diff_matrix * (2 * np.pi) / fs  # convert to radians
    diff_matrix = 100 * diff_matrix / ref_freqs[:, np.newaxis]  # convert to relative error

    # Compute minimum possible error for each element in ref_sort, discard the "num_misses" largest errors
    err_min_per_freq = np.sort(np.min(diff_matrix, axis=1))[::-1][num_misses:]

    # err_avg = np.mean(err_min_per_freq ** 2)  # mean squared error (MSE)
    err_avg = np.mean(err_min_per_freq)  # mean absolute error (MAE)

    res = {'Padded FFT' : err_avg}

    return res


def evaluate_crb(crb_infos, true_freqs, fs):
    """
    Evaluate the CRB for the frequency estimation.

    Stoica, P. G., Moses, R., Stoica, P., & Moses, R. L. (2005). Spectral analysis of signals. Pearson, Prentice Hall.
    Equation (4.3.9).
    """

    if not crb_infos:
        return {}

    # Eq. (4.3.9) from Stoica and Moses (2005)
    crb_vec = (6 * crb_infos['white_noise_var_crb'] / crb_infos['N'] ** 3) * (1. / crb_infos['sin_amp_squared_crb'])

    # Convert to relative error
    # Square coefficient as CRB is variance of estimator (Var(aX + b) = a^2 Var(X))
    crb_vec = crb_vec * (100. * fs / (2. * np.pi * true_freqs)) ** 2

    # average over estimated frequencies
    crb_avg = np.mean(crb_vec)

    # Square root of the average CRB (refers to MAE, not MSE/variance)
    crb_avg = np.sqrt(crb_avg)

    return {'CRB' : crb_avg}


def log_intermediate_results(plot_args_bf, plot_args_freq_est, varying_param_values, plot_sett):
    if plot_args_bf:
        if 'snrseg_time' in plot_args_bf['metrics_list']:
            Evaluator.print_final_results_table(plot_args_bf['result_by_metric']['snrseg_time'],
                                                plot_args_bf['algorithms'], varying_param_values)
        if plot_sett['intermediate_results']:
            fig_bf = pl.plot_results(**plot_args_bf, **plot_sett)

    if plot_args_freq_est:
        for metric_name, results in plot_args_freq_est['result_by_metric'].items():
            print(f"Frequency estimation results for {metric_name}:")
            Evaluator.print_final_results_table(results, plot_args_freq_est['algorithms'], varying_param_values)
        if plot_sett['intermediate_results']:
            fig_freq_est = pl.plot_results(**plot_args_freq_est, **plot_sett)

    return
