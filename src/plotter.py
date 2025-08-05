import copy
import warnings
from pathlib import Path
import librosa
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from prettytable import PrettyTable

from . import utils as u
from . import config

num_reps = 3
# List the markers in a way that consecutive markers are not too similar (for example, < and > are too similar and should be far apart)
markers = ['o', 'x', 's', 'D', '*', 'v', 'p', '>', 'h', '^', '|', '<', '+', '_'] * num_reps

line_styles = ['-', '--', ':', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':'] * num_reps
colors = ['tab:blue', 'tab:orange', 'tab:brown', 'tab:red', 'tab:brown', 'tab:purple',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] * num_reps


# colors = ['tab:brown', 'tab:brown',
#           'tab:pink', 'tab:gray',
#           'tab:olive', 'tab:cyan'] * num_reps

# TODO: add entry for 'standard error' in legend


def check_if_log_scale(arr):
    try:
        # Check if the array approximately follows a log scale, e.g. 1, 10, 100, 1000, ...
        # if arr.size < 2 or np.any(arr <= 0):
        if arr.size < 2:
            return False

        # Instead of checking each element, compute all ratios and check their mean
        ratios = []
        for idx, _ in enumerate(arr[:-1]):
            if np.abs(arr[idx]) > 1e-12:
                ratios.append(arr[idx + 1] / (1e-15 + arr[idx]))
        ratios = np.array(ratios)

        looks_like_log_distribution = True
        if np.mean(ratios) < 1.8:
            looks_like_log_distribution = False

    except Exception as e:
        print(f"Exception in check_if_log_scale: {e}")
        looks_like_log_distribution = False

    return looks_like_log_distribution


def decapitalize(s):
    if not s:  # check that s is not empty string
        return s
    if s[1].isupper():  # if second letter is uppercase, skip (do not decapitalize SNR to sNR!)
        return s
    return s[0].lower() + s[1:]


def plot_results_single_metric(ax, varying_param_values, result_single_metric, metric_display_name, algorithms,
                               name_varying_param, add_date_to_title=True, show_title=True, show_legend=True):
    """
    Plot the results of a single metric vs a varying parameter.

    :param ax: Matplotlib axis
    :param varying_param_values: List of values of the varying parameter (e.g., [0, 1, 2, 3, 4, 5])
    :param result_single_metric: Numpy array of shape (num_algorithms, num_varying_param_values, 3)
    :param metric_display_name: Display name of the metric (e.g., 'STOI')
    :param algorithms: List of algorithms (e.g., ['Noisy', 'MWF [blind]', 'cMWF [blind]', 'MWF [oracle]', ...])
    :param name_varying_param: Name of the varying parameter (e.g., 'f0_err_percent'). This will be displayed on the x-axis.
    :param add_date_to_title: If True, add the current date and time to the title of the plot.
    :return:
    """
    x_values_raw = np.array(varying_param_values)
    x_values = convert_varying_param_values_to_display(x_values_raw, name_varying_param)

    num_max_x_ticks = 16
    num_x_ticks = min(num_max_x_ticks, len(x_values))

    if add_date_to_title:
        font_size = 'medium'
        font_size_ticks_labels = 'medium'
        legend_font_size = 'small'
        title_font_size = 'medium'
        date_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        title = f'{metric_display_name} vs {decapitalize(name_varying_param)} ({date_time})'
    else:
        # # Adjust font sizes
        title_font_size = 9
        font_size = 7
        font_size_ticks_labels = 7
        legend_font_size = 7
        title = f'{metric_display_name} vs {decapitalize(name_varying_param)}'

        # Old font sizes
        # font_size = 'large'
        # font_size_ticks_labels = 'large'
        # legend_font_size = 'medium'
        # title_font_size = 'x-large'
        # title = f'{metric_display_name} vs {decapitalize(name_varying_param)}'

    # Algorithms: ['Noisy', 'MWF [blind] ', 'cMWF [blind] (prop.)', 'MWF [oracle] ', 'cMWF [oracle] (prop.)', 'MWF [super-oracle] ', 'cMWF [super-oracle] (prop.)']

    for algo_index, algo in enumerate(algorithms):
        res_single_metric_single_algo = result_single_metric[algo_index]

        algo_lower = algo.lower()

        line_style = 'solid'
        line_width = 1.2
        marker = markers[algo_index]
        if get_variant_display_name('oracle') in algo_lower:  # should be checked first because + is part of ++
            line_style = 'dotted'
            line_width = 0.8
        elif get_variant_display_name('semi-oracle') in algo_lower:
            line_style = 'dashed'
            line_width = 0.9

        # line_width = line_width * 1.2  # increase the line width

        if 'pfcmvdr' in algo_lower and 'wl' not in algo_lower:
            color = '#ab2c32'
            marker = 'p'

        elif 'cmvdr' in algo_lower and 'wl' not in algo_lower:
            color = '#dc2f02'  # red
            marker = 's'
            if get_variant_display_name('oracle') in algo_lower:  # should be checked first because + is part of ++
                color = '#faa307'  # yellow
                marker = '^'
            elif get_variant_display_name('semi-oracle') in algo_lower:
                color = '#e85d04'  # orange
                marker = 'D'
        elif 'clcmv' in algo_lower:
            color = 'tab:blue'
        elif 'cmwf' in algo_lower:
            color = '#dc2f02'  # red
            marker = 's'
            if get_variant_display_name('oracle') in algo_lower:  # should be checked first because + is part of ++
                color = '#faa307'  # yellow
                marker = '^'
            elif get_variant_display_name('semi-oracle') in algo_lower:
                color = '#e85d04'  # orange
                marker = 'D'
        elif 'mwf' in algo_lower:
            color = '#68489C'  # 'tab:blue'
            marker = 'x'
            if get_variant_display_name('oracle') in algo_lower:  # should be checked first because + is part of ++
                color = 'tab:cyan'
                marker = '*'
            elif get_variant_display_name('semi-oracle') in algo_lower:
                color = 'tab:blue'
                marker = '+'
        elif 'mvdr' in algo_lower and 'wl' not in algo_lower:
            color = '#68489C'  # 'tab:blue'
            marker = 'x'
            if get_variant_display_name('oracle') in algo_lower:  # should be checked first because + is part of ++
                color = 'tab:cyan'
                marker = '*'
            elif get_variant_display_name('semi-oracle') in algo_lower:
                color = 'tab:blue'
                marker = '+'
        else:
            # Use same colour for same variant, e.g. for MWF [blind] and cMWF [blind]
            color = colors[algo_index]
            if 'cmwf' in algo_lower:
                color = colors[algo_index - 1]  # this skips some colors

        # if last dimension is not unitary, it contains (mean, standard deviation = std). Otherwise, only mean is present.
        # Plot the std as a shaded area around the mean
        ax.fill_between(x_values, res_single_metric_single_algo[:, 1],
                        res_single_metric_single_algo[:, 2],
                        facecolor=color, alpha=0.1)

        # Plot the mean
        ax.plot(x_values, res_single_metric_single_algo[:, 0], label=algo,
                linestyle=line_style, marker=marker, markersize=5,
                markeredgecolor=color, markerfacecolor='none', markeredgewidth=0.5,
                linewidth=line_width, c=color)

    # If varying_param_values have a log scale (e.g., f0_err_percent: 0, 1e-4, 1e-2, 1e-1, ...) then use log scale
    use_log_scale = check_if_log_scale(np.array(x_values))
    lin_thresh = x_values[1] - x_values[0] if len(x_values) > 1 else 1
    # lin_thresh = np.median(np.diff(x_values))  # Use median spacing to balance uniformity
    if use_log_scale:
        ax.set_xscale('symlog', linthresh=lin_thresh)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys(), fontsize=legend_font_size, ncol=1)
        # Put legend below the plot
        # num_columns = 3 if len(algorithms) > 4 else 2
        num_columns = 4
        ax.get_figure().legend(by_label.values(), by_label.keys(), fontsize=legend_font_size,
                               ncol=num_columns, loc='outside lower center')

    ax.set_xlabel(f'{name_varying_param}', fontsize=font_size, labelpad=2)
    ax.set_ylabel(metric_display_name, fontsize=font_size, labelpad=2)

    # x, y ticks and grid
    if ax.get_xscale() == 'log':
        x_locator = tck.SymmetricalLogLocator(base=10, linthresh=lin_thresh)
        x_minor_locator = None
        y_minor_locator = tck.AutoMinorLocator(4)
    else:
        if num_x_ticks < 8:  # only a few x-ticks, so show all of them
            x_locator = tck.FixedLocator(x_values)
            x_minor_locator = None
            y_minor_locator = tck.AutoMinorLocator(2)
        else:
            x_locator = tck.MaxNLocator(num_x_ticks - 1, min_n_ticks=4)
            x_minor_locator = tck.AutoMinorLocator(4)
            y_minor_locator = tck.AutoMinorLocator(2)

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, fontsize=font_size_ticks_labels)
    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
    ax.tick_params(axis='both', labelsize=font_size_ticks_labels, pad=2)
    ax.grid(which='both')

    # Change minor ticks to show every 5. (20/4 = 5)
    if x_minor_locator is not None:
        ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    ax.grid(which='major', color='#CCCCCC')
    ax.xaxis.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.3)
    ax.yaxis.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)

    if name_varying_param == 'est_error_snr':  # flip the x-axis
        ax.invert_xaxis()

    if show_title:
        ax.set_title(title, fontsize=title_font_size)

    return ax


def plot_results(varying_param_values, result_by_metric, metrics_list, algorithms, parameter_to_vary='',
                 save_plots=False, separately=False, show_date_plots=True, show_title=True, show_legend=True,
                 use_tex=False, force_no_plots=False, f0_spectrogram=False, target_folder_path=Path('figs'),
                 **kwargs):
    """ Plot the results of multiple metrics vs a varying parameter. """

    if not result_by_metric or not metrics_list or not algorithms:
        return None

    plot_area_size = 3.5

    # Get the display names of the metrics, e.g., 'STOI', ...
    metrics_list_display_name = [get_metric_display_name_with_type(metric, is_for_export=use_tex)
                                 for metric in metrics_list]

    if kwargs.get('is_relative_result', False):
        if use_tex:  # for export, we use Latex, so the delta symbol (if present) is replaced by the command $\Delta$
            metrics_list_display_name = [f"$\\Delta${metric_name}" for metric_name in metrics_list_display_name]
        else:
            metrics_list_display_name = [f"Δ{metric_name}" for metric_name in metrics_list_display_name]
    num_plots = len(metrics_list_display_name) if not separately else 1
    name_varying_param = get_parameter_display_name(parameter_to_vary, use_tex=use_tex)

    if save_plots:
        target_folder_path.mkdir(parents=True, exist_ok=True)

    figs = []
    if separately:
        for idx, metric in enumerate(result_by_metric.keys()):
            x_size = u.get_plot_width_double_column_latex() / 2  # two figs should fit one column (4 figs in one row)
            fig = plt.figure(figsize=(x_size, 0.8 * x_size), dpi=150, constrained_layout=True)
            ax = fig.subplots(nrows=1, ncols=1, squeeze=True)

            metric_disp_name = metrics_list_display_name[idx]
            ax = plot_results_single_metric(ax, varying_param_values, result_by_metric[metric],
                                            metric_disp_name, algorithms, name_varying_param,
                                            add_date_to_title=show_date_plots, show_title=show_title,
                                            show_legend=show_legend)

            if 'forced_ranges' in kwargs:
                if metric in kwargs['forced_ranges']:
                    ax.set_ylim(kwargs['forced_ranges'][metric])

            if save_plots:
                name_second_part = make_dir_get_saving_path(metric_disp_name, name_varying_param)
                u.savefig(fig, target_folder_path / (name_second_part + '.pdf'))
            figs.append(fig)
            fig.show()

    else:  # print all metrics in one figure (debugging)
        if num_plots < 3:
            fig = plt.figure(figsize=(1 + plot_area_size * 0.9, 1 + num_plots * plot_area_size), dpi=150,
                             constrained_layout=True)
            axs = fig.subplots(nrows=num_plots, ncols=1, squeeze=True, sharex=True)
        else:
            fig = plt.figure(figsize=(1 + 2 * plot_area_size, 1 + num_plots * plot_area_size / 2), dpi=150,
                             constrained_layout=True)
            num_rows = int(np.ceil(num_plots / 2))
            axs = fig.subplots(nrows=num_rows, ncols=2, squeeze=True)

        # Convert to numpy array to avoid errors when there is only one metric
        axs = np.array(axs, ndmin=1, dtype=object).flatten()

        for idx, metric in enumerate(result_by_metric.keys()):
            metric_disp_name = metrics_list_display_name[idx]
            axs[idx] = plot_results_single_metric(axs[idx], varying_param_values, result_by_metric[metric],
                                                  metric_disp_name, algorithms, name_varying_param,
                                                  add_date_to_title=show_date_plots, show_title=show_title,
                                                  show_legend=show_legend)

            if 'forced_ranges' in kwargs:
                if metric in kwargs['forced_ranges']:
                    axs[idx].set_ylim(kwargs['forced_ranges'][metric])

        # If there are 3 plots, the last one is empty, so remove it
        if num_plots == 3:
            fig.delaxes(axs[-1])

        if save_plots:
            name_second_part = make_dir_get_saving_path('metrics', name_varying_param)
            u.savefig(fig, target_folder_path / (name_second_part + '.pdf'))
        figs.append(fig)
        fig.show()

    return figs


def visualize_all_results(results_data_type_, plot_sett_, cfg, plot_db=False, print_summary=False,
                          target_path_figs_=Path('figs'), print_full_table=True):
    plot_sett_['destination'] = plot_sett_['destination_final_results']

    plot_sett_ = config.ConfigManager.get_plot_settings(plot_sett_)
    plot_sett_['show_legend'] = False if plot_sett_['destination'] == 'paper' else True
    plot_sett_['show_title'] = False if plot_sett_['destination'] == 'paper' else True
    plot_sett_['target_folder_path'] = target_path_figs_
    u.set_plot_options(use_tex=plot_sett_['use_tex'])
    plot_sett_['save_plots'] = False if cfg['num_montecarlo_simulations'] <= 1 else plot_sett_['save_plots']

    # plot_sett_['forced_ranges'] = {'sisdr_time': [-1, 27.5]}

    figs = []
    for parameter_to_vary_, plot_args_ in results_data_type_.items():

        if not plot_args_:
            continue

        # Print in table format the same data we would have in the plot
        if print_full_table:
            # Create and print the table
            table = PrettyTable()
            pretty_param_name = get_parameter_display_name(plot_args_['parameter_to_vary'])
            table.field_names = ([pretty_param_name] +
                                 [str(x) for x in plot_args_['algorithms']])

            print(f"\nResults averaged for different parameter {parameter_to_vary_}:")
            for metric, result_array in zip(plot_args_['metrics_list'], plot_args_['result_by_metric'].values()):
                print(f"{metric = }")
                varyings_values = convert_varying_param_values_to_display(plot_args_['varying_param_values'], pretty_param_name)
                for idx_var in range(result_array.shape[1]):
                    row_data = result_array[:, idx_var, 0]
                    table.add_row([varyings_values[idx_var]] + row_data.tolist())
            table.float_format = ".2"
            table.align = "r"

            print(table)

        # Average results over all variations of the parameter, so that we have a table with the average results
        if print_summary:
            # Create and print the table
            table = PrettyTable()
            table.field_names = ["Algorithms"] + [str(x) for x in plot_args_['algorithms']]

            # Single row, each cell is the averaged result over varying_param_values for a single algorithm
            print(f"\nResults averaged over parameter {parameter_to_vary_}:")
            for metric, result_array in zip(plot_args_['metrics_list'], plot_args_['result_by_metric'].values()):
                # shape of result_array: (num_algorithms, num_varying_param_values, 3)
                # Only select mean (ignore +/- 95% confidence interval)
                result_array_avg = np.mean(result_array[..., 0], axis=1)
                row_data = result_array_avg
                table.add_row([metric] + row_data.tolist())
            table.float_format = ".3"
            table.align = "r"

            print(table)

        if plot_sett_['destination'] == 'none':
            return None

        if plot_db:
            plot_args_['result_by_metric'] = plot_args_['result_by_metric_db']
            for idx, metric in enumerate(plot_args_['metrics_list']):
                if 'freq-err-mae' in metric:
                    plot_args_['metrics_list'][idx] = 'freq-err-mae-db'

        figs = plot_results(**plot_args_, **plot_sett_)
        for fig_ in figs:
            fig_.show()

    if plot_sett_['destination'] != 'debug' and plot_sett_['destination'] != 'none' and plot_sett_['save_plots']:
        # Save cfg as a yaml file to plot_sett_['target_folder_path']
        # To save the configuration file (dict) as a yaml file, we do:
        Path(plot_sett_['target_folder_path']).mkdir(parents=True, exist_ok=True)
        path_config_yaml_ = plot_sett_['target_folder_path'] / 'config.yaml'
        config.write_configuration(cfg, path_config_yaml_)

    return figs


def make_dir_get_saving_path(metric_disp_name, name_varying_param):
    name_second_part = f'{metric_disp_name}_vs_{name_varying_param}'
    to_replace = {' ': '_', '=': '', ',': '', '.': '', '[': '', ']': '', '(': '', ')': '',
                  '$': '', '\\': '', '%': 'percent', '{': '', '}': '', '#': 'num', '\\Delta': 'Delta'}

    for key, value in to_replace.items():
        name_second_part = name_second_part.replace(key, value)

    if not Path('figs').is_dir():
        Path('figs').mkdir()

    return name_second_part


def get_display_name_beamformer(first_name_internal):
    postfix_str = ''
    if first_name_internal == 'mvdr':
        first_name_display = 'MVDR'
    elif first_name_internal == 'mwf':
        first_name_display = 'MWF'
    elif first_name_internal == 'cmwf':
        first_name_display = 'cMWF'
        postfix_str = '(prop.)'
    elif first_name_internal == 'cmvdr':
        first_name_display = 'cMVDR'
        postfix_str = '(prop.)'
    elif first_name_internal == 'cmvdr-wl':
        first_name_display = 'cMVDR-WL'
        postfix_str = '(prop.)'
    elif first_name_internal == 'clcmv':
        first_name_display = 'cLCMV'
        postfix_str = '(prop.)'
    elif first_name_internal == 'noisy':
        first_name_display = 'Noisy'
    elif first_name_internal == 'wet':
        # first_name_display = 'Reverberant (=wet)'
        first_name_display = 'Target'
    elif first_name_internal == 'early':
        first_name_display = 'Target (at mic 0)'
    elif first_name_internal == 'noise':
        first_name_display = 'Noise'
    elif first_name_internal == 'noise_cov_est':
        first_name_display = 'Noise (for cov. est.)'
    elif first_name_internal == 'noise_freq_est':
        first_name_display = 'Noise (for freq. est.)'
    else:
        warnings.warn(f"Algorithm name {first_name_internal} not found in get_display_name_beamformer")
        first_name_display = first_name_internal

    return first_name_display, postfix_str


def get_variant_display_name(variant_internal):
    if variant_internal == 'oracle':
        variant_display = '++'
    elif variant_internal == 'semi-oracle':
        variant_display = '+'
    elif variant_internal == 'blind':
        variant_display = ''
    elif variant_internal == 'rank1':
        # variant_display = '[rank-1]'
        variant_display = ''
    elif variant_internal == '':
        variant_display = ''
    else:
        warnings.warn(
            f"Algorithm variant {variant_internal} not found in get_display_name. Returning empty string.")
        variant_display = ''

    return variant_display


def get_display_name(algo_name_internal):
    # Convert the internal name of the algorithm to the display name
    # For example, 'MVDR_oracle' -> 'MVDR (Narrowband) [oracle]'
    # Another example: 'cmwf_semi-oracle' -> 'MWF (Cyclic) (prop.) [semi-oracle]'

    # return algo_name_internal
    algo_name_internal = algo_name_internal.lower()

    try:
        first_name_internal, variant_internal = algo_name_internal.split('_')
    except ValueError:
        first_name_internal = algo_name_internal
        variant_internal = ''

    first_name_display, postfix_str = get_display_name_beamformer(first_name_internal)
    variant_display = get_variant_display_name(variant_internal)

    final_name = first_name_display
    if postfix_str != '':
        final_name = f'{first_name_display}{variant_display} {postfix_str}'
    elif variant_display != '':
        final_name = f'{first_name_display}{variant_display}'

    return final_name


def get_metric_display_name(metric):
    """
    # metrics_list = ['stoi', 'pesq', 'rmse', 'fwsnrseg', 'snr_fullband']
    # metrics_list_display_name = ['STOI', 'PESQ', 'rmse [dB]', 'fwSNRseg [dB]', 'SNR']
    """
    metric = metric.lower()
    if metric == 'stoi':
        return 'STOI'
    elif metric == 'pesq':
        return 'PESQ'
    elif metric == 'rmse':
        return 'RMSE [dB]'
    elif metric == 'fwsnrseg':
        return 'fwSNRseg [dB]'
    elif metric == 'snrseg':
        return 'SNRseg [dB]'
    elif metric == 'snr_fullband':
        return 'SNR'
    elif metric == 'mmse':
        return 'MMSE [dB]'
    elif metric == 'mae':
        return 'MAE [dB]'
    elif metric == 'sinr':
        return 'SINR [dB]'
    elif metric == 'freq-err-mae':
        return '$E_{\\%}$'
    elif metric == 'freq-err-mae-db':
        return '$E_{\\text{dB}}$'
    elif metric == 'num-misses':
        return 'Num. misses'
    elif metric == 'sisdr':
        return 'SI-SDR [dB]'
    else:
        warnings.warn(f"Light warning: metric {metric} not found in get_metric_display_name")
        return metric


def get_metric_display_name_with_type(metric_input, is_for_export=False):
    """
    # metrics_list = ['stoi', 'pesq', 'rmse', 'fwsnrseg', 'snr_fullband']
    # metrics_list_display_name = ['STOI', 'PESQ', 'rmse [dB]', 'fwSNRseg [dB]', 'SNR']
    """
    metric_input = metric_input.lower()
    metric_input_split = metric_input.split('_')
    if len(metric_input_split) == 2 and (metric_input_split[1] == 'time' or metric_input_split[1] == 'freq'):
        metric, metric_type = metric_input_split
    else:
        metric = metric_input
        metric_type = ''

    if metric_type != 'time' and metric_type != 'freq' and metric_type != '':
        raise ValueError(f"Metric type {metric_type} not recognized. It should be either 'time' or 'freq' or ''.")

    metric_display = get_metric_display_name(metric)
    if 'fwSNRseg' not in metric_display and 'SNRseg' not in metric_display:
        # don't report (time) or (freq) in the exported plots
        if is_for_export or metric_type == '':
            return f'{metric_display}'
        else:
            return f'{metric_display} ({metric_type})'
    else:
        return f'{metric_display}'


def get_parameter_display_name(parameter_to_vary, use_tex=False):
    if parameter_to_vary == 'snr_db_dir_noise' or parameter_to_vary == 'noise|snr_db_dir':
        return 'iSNR [dB]'
    elif parameter_to_vary == 'noise|harmonic_correlation':
        if not use_tex:
            return 'Noise harmonic correlation'
        else:
            return r'Noise corr. $\beta$'
    elif parameter_to_vary == 'cyclic|harmonic_threshold':
        return "Coherence threshold"
    elif parameter_to_vary == 'cyclic|P_max':
        if use_tex:
            return r'\# mods $C_{\text{max}}$'
        else:
            return 'Num. modulations'
    elif parameter_to_vary == 'cyclic|f0_err_percent':
        return 'Frequency est. error [\\%]'
        # return 'Bias error in fun. freq. $\\Delta \\alpha_1$ [\\%]'
        # return "Error in fundamental frequency $\Delta$"
    elif parameter_to_vary == 'est_error_snr':
        return 'estimation error SNR [dB]'
    elif parameter_to_vary == 'M':
        if use_tex:
            return r'\# mics $M$'
        else:
            return 'Num. of mic. M'
    elif parameter_to_vary == 'duration_approx_seconds':
        return "Signal length [s]"
    elif parameter_to_vary == 'chunk_len':
        return "Covariance estimation time [s]"
    elif parameter_to_vary == 'freq_range_cyclic':
        return "max freq. cyclic [Hz]"
    elif parameter_to_vary == 'target_angle':
        return "Target angle [deg]"
    elif parameter_to_vary == 'rir_specs|rt60':
        return "RT60 [ms]"
    elif parameter_to_vary == 'noise|inharmonicity_percentage':
        return "Noise inharmonicity [\\%]"
    elif parameter_to_vary == 'target|inharmonicity_percentage':
        return "Target inharmonicity [\\%]"
    elif parameter_to_vary == 'beamforming|loadings|mwf':
        return "MWF loading"
    elif parameter_to_vary == 'beamforming|loadings|mvdr':
        return "MVDR max. loading"
    elif parameter_to_vary == 'cov_estimation|noise_cov_est_len_seconds':
        return "Noise cov. est. time [s]"
    elif parameter_to_vary == 'harmonics_est|max_len_seconds':
        return "Harmonic freq. est. time [s]"
    elif parameter_to_vary == 'harmonics_est|snr_db_relative':
        return "Harmonic freq. est. SNR [dB]"
    elif parameter_to_vary == 'harmonics_est|max_num_harmonics_peaks':
        return "Max. num. harmonic peaks"
    elif parameter_to_vary == 'harmonics_est|nfft':
        return "Harmonic freq. est. FFT size"
    elif parameter_to_vary == 'stft|nfft':
        return "FFT size"
    else:
        print(f"Light warning: parameter {parameter_to_vary} not found in get_parameter_display_name")
        return parameter_to_vary


def convert_varying_param_values_to_display(x_values_raw, name_varying_param_display):

    display_name = lambda x : get_parameter_display_name(x)

    if name_varying_param_display == display_name('chunk_len'):
        return x_values_raw / 16000  # convert to seconds
    elif name_varying_param_display == display_name('freq_range_cyclic'):
        return [x[1] for x in x_values_raw]
    elif name_varying_param_display == display_name('target_angle'):
        # convert from ['000', '015', '030', '045', '060', '075', '090', '270', '285', '300', '315', '330', '345']
        # to [0, 15, 30, 45, 60, 75, 90, -90, -75, -60, -45, -30, -15]
        # So we need to handle the conversion of values larger than 180 to negative values
        x_values_raw = [int(x) for x in x_values_raw]
        x_values = []
        for x in x_values_raw:
            if x > 180:
                x_values.append(x - 360)
            else:
                x_values.append(x)
        return x_values
    elif name_varying_param_display == display_name('rir_specs|rt60'):
        # "0.160s" --> 160
        return [int(x.split('s')[0].split('.')[1]) for x in x_values_raw]
    elif name_varying_param_display == display_name('noise|inharmonicity_percentage'):
        return [float(x) for x in x_values_raw]
    elif name_varying_param_display == display_name('target|inharmonicity_percentage'):
        return [float(x) for x in x_values_raw]
    elif name_varying_param_display == display_name('beamforming|loadings|mwf'):
        return [float(x[0]) for x in x_values_raw]  # 0 to display minimum value and 1 to display maximum value
    elif name_varying_param_display == display_name('beamforming|loadings|mvdr'):
        return [float(x[1]) for x in x_values_raw]  # 0 to display minimum value and 1 to display maximum value
    return x_values_raw


def plot_spectrograms(signals_dict, freqs_hz, max_displayed_frequency_bin=10000, max_time_frames=500, save_figs=False,
                      slice_chunks=None, delta_t=None, suptitle=''):
    # if plot_spectrograms:
    # Prepare the settings for plotting
    xy_labels = ('Time frame', 'Frequency bin')
    amp_range = (-130, 0)
    plt_sett = {'xy_label': xy_labels, 'amp_range': amp_range, 'normalized': False, 'show_figures': False}
    # signals_to_skip = ['noise', 'noise_cov_est', 'wet', 'mvdr_semi-oracle', 'cmvdr_semi-oracle']
    signals_to_skip = ['noise', 'noise_cov_est', 'wet']
    font_size = 'large'

    existing_figs = []
    im = None

    signals_dict_spectrogram = {key: value for key, value in signals_dict.items() if key not in signals_to_skip}

    if all([isinstance(x, dict) for x in signals_dict_spectrogram.values()]):
        signals_display_names = [x['display_name'] for x in signals_dict_spectrogram.values()]
    else:
        signals_display_names = [x for x in signals_dict_spectrogram.keys()]

    # Normalize the spectrograms so that the maximum value is 1, but the relative values are preserved
    max_amplitude = -np.inf
    for signal in signals_dict_spectrogram.values():
        signal_stft = signal['stft'] if 'stft' in signal else signal
        max_amplitude = max(max_amplitude, np.max(np.abs(signal_stft)))

    max_displayed_frequency_bin = min(max_displayed_frequency_bin, len(freqs_hz))
    local_trans = lambda x: np.abs(x)[:max_displayed_frequency_bin, :max_time_frames] / max_amplitude

    num_frames = 0
    for sig_name, signal in signals_dict_spectrogram.items():
        signal_stft = signal['stft'] if 'stft' in signal else signal
        signal_display_name = signal['display_name'] if 'display_name' in signal else sig_name
        temp = local_trans(signal_stft)
        ff = u.plot_matrix(temp, title=signal_display_name, **plt_sett)
        existing_figs.append(ff)
        num_frames = temp.shape[-1]

    # # Combine the SCFs plots into a single figure
    plot_area_size = 5
    fig = plt.figure(figsize=(1 + plot_area_size, plot_area_size * len(signals_dict_spectrogram) / 4), dpi=150)

    # Set the frequency ticks
    num_y_ticks_approx = 5
    delta_freq_bins = max_displayed_frequency_bin // num_y_ticks_approx
    y_ticks = np.arange(0, max_displayed_frequency_bin, delta_freq_bins)
    y_tick_labels = [f"{np.abs(freqs_hz[kk] / 1000):.1f}" for kk in
                     range(0, max_displayed_frequency_bin, delta_freq_bins)]

    # Set the x-axis ticks to show time in seconds
    x_ticks = np.array([])
    x_tick_labels = []
    if delta_t is not None and num_frames > 0:
        num_x_ticks_approx = 4
        delta_frames = num_frames // num_x_ticks_approx
        x_ticks = np.arange(0, num_frames, delta_frames)
        x_tick_labels = [f"{delta_t * x_tick:.1f}" for x_tick in x_ticks]

    num_rows = int(np.ceil(len(signals_display_names) / 2))
    for ii, (existing_fig, title) in enumerate(zip(existing_figs, signals_display_names)):
        ax = fig.add_subplot(num_rows, 2, ii + 1)
        ax.set_yticks(y_ticks, y_tick_labels, fontsize=font_size)

        if x_ticks.size > 0:
            ax.set_xticks(x_ticks, x_tick_labels, fontsize=font_size)
            ax.set_xlabel('Time [s]', fontsize=font_size)
        else:
            plt.tick_params(reset=True, axis='x', labelsize=font_size)

        im = u.fig_to_subplot(existing_fig, title, ax,
                              xlabel='Time [s]',
                              ylabel='Frequency [kHz]',
                              font_size=font_size, title_font_size=font_size)

    for ax, sig_name in zip(fig.get_axes(), signals_dict_spectrogram.keys()):
        ax.label_outer()
    #     if slice_chunks is not None and sig_name == 'early':
    #         for chunk in slice_chunks:
    #             ax.axvline(chunk.stop, color='g', linestyle='-', linewidth=0.5)

    if im:
        cl = fig.colorbar(im, ax=fig.get_axes(), orientation='vertical', location='right', pad=0.02, shrink=0.8,
                          aspect=40)
        cl.set_label('Normalized amplitude [dB]', fontsize=font_size)

    if suptitle:
        fig.suptitle(suptitle, fontsize=font_size)

    fig.show()
    if save_figs:
        u.savefig(fig, Path('figs') / 'spectrograms.pdf')

    for f in existing_figs:
        plt.close(f)

    return fig


# def plot_results_outer(parameter_to_vary, varying_param_values, metrics_list_display_name,
#                        result_by_metric, algorithms,
#                        save_figs=False, plot_separately=False, show_date_plots=True):
#     plt_args = {'varying_param_values': varying_param_values, 'result_by_metric': result_by_metric,
#                 'metrics_list_display_name': metrics_list_display_name, 'algorithms': algorithms,
#                 'name_varying_param': get_parameter_display_name(parameter_to_vary),
#                 'save_figs': save_figs,
#                 'separately': plot_separately, 'show_date': show_date_plots}
#     plot_results(**plt_args)


def plot_waveforms(signals_dict, dft_props, alpha_list, num_chunks, slice_bf_list, save_plots=False):
    plot_list = [signals_dict[key]['time'] for key in signals_dict.keys() if key != 'noise']
    signals_display_names = [x['display_name'] for x in signals_dict.values() if x['display_name'] != 'Noise']
    fs = dft_props['fs']

    f1 = u.plot(plot_list, show_fig=False, titles=signals_display_names, fs=fs, subplot_height=1.2)

    if 1 <= num_chunks <= 30:

        for idx_ax, ax in enumerate(f1.get_axes()):
            ax.label_outer()
            num_samples_chunk_prev = 0
            for idx_chunk in range(num_chunks):  # Draw dashed lines to indicate the start of each chunk
                num_frames_chunk = slice_bf_list[idx_chunk].stop - slice_bf_list[idx_chunk].start
                num_samples_chunk = (num_frames_chunk - 1) * dft_props['r_shift_samples'] + dft_props['nfft']
                chunk_end = num_samples_chunk_prev + num_samples_chunk
                chunk_middle = num_samples_chunk_prev + num_samples_chunk // 2
                ax.axvline(chunk_end, color='g', linestyle='--', linewidth=0.5)

                # Write chunk number on the plot. For each chunk, also display a column (histogram like) that shows the number
                # of elements in alpha_list for that chunk.
                if idx_ax == 0:
                    ax.text(chunk_middle, -0.92, f"{idx_chunk}", fontsize=8, color='r')
                    # ax.text(idx_chunk * chunk_len + .6, -0.92, f"{idx_chunk}", fontsize=8, color='r')

                if idx_ax == len(f1.get_axes()) - 1:
                    # Draw the histogram like column on the last plot
                    bar_width = num_samples_chunk / 2
                    last_plot = f1.get_axes()[-1]
                    if len(alpha_list[idx_chunk]) > 1:
                        P_max = len(alpha_list[idx_chunk]) - 1
                        last_plot.bar(x=chunk_middle, height=(len(alpha_list[idx_chunk]) - 1) / P_max,
                                      width=bar_width,
                                      bottom=-.95,
                                      color='g', alpha=0.5)
                        if 1 <= num_chunks <= 30:
                            last_plot.text(chunk_middle, -0.92, f"{len(alpha_list[idx_chunk])}", fontsize=8,
                                           color='b')

                num_samples_chunk_prev += num_samples_chunk

    f1.show()

    if save_plots:
        u.savefig(f1, Path('figs') / 'waveforms.pdf')


def plot_rmse_before_average(signals, dft_props, ref_clean='wet_rank1', which_axis='time', max_freq_hz=3000,
                             plot_diff_between_algos=False, benchmark='mvdr_blind'):
    # Plot the RMSE between the reference signal and the signals. Average over "which_axis" axis.

    # skip_list = ['noise', 'noise_cov_est', 'noisy', 'wet', ref_name, 'mwf_blind', 'cmwf_blind']
    skip_list = ['noise', 'noise_cov_est', 'noisy', 'wet', ref_clean]

    K_nfft_real = dft_props['nfft_real']

    if max_freq_hz < dft_props['fs'] / 2:
        K_nfft_real = int(max_freq_hz / dft_props['fs'] * dft_props['nfft'])

    errors_dict_raw = {}
    for sig_name, signal_dict in signals.items():
        if sig_name in skip_list:
            continue

        signal = signal_dict['stft'][:K_nfft_real]
        ref = signals[ref_clean]['stft'][:K_nfft_real]

        if which_axis == 'time':
            errors_dict_raw[sig_name] = np.sqrt(np.mean(np.abs(ref - signal) ** 2, axis=0))
        elif which_axis == 'freq':
            errors_dict_raw[sig_name] = np.sqrt(np.mean(np.abs(ref - signal) ** 2, axis=1))
        else:
            raise ValueError(f"which_axis should be either 'time' or 'freq'. Got {which_axis} instead.")

    errors_dict = copy.deepcopy(errors_dict_raw)
    if plot_diff_between_algos and benchmark in errors_dict:
        for algo_index, (algo, error) in enumerate(errors_dict.items()):
            if algo != benchmark:
                errors_dict[algo] = errors_dict_raw[algo] - errors_dict_raw[benchmark]
        errors_dict.pop(benchmark, None)

    # Skip if plotting all zeros
    all_zeros = False
    for error in errors_dict.values():
        if np.allclose(error, 0):
            all_zeros = True
            break

    if all_zeros:
        warnings.warn("Skip error per freq plot, all zeros")
        return

    # Plot the RMSE
    fig = plt.figure(figsize=(5, 4), dpi=300)
    ax = fig.subplots(nrows=1, ncols=1, squeeze=True)

    both_oracle_and_blind_present = any(['oracle' in algo.lower() for algo in errors_dict.keys()]) and any(
        ['blind' in algo.lower() for algo in errors_dict.keys()])

    min_val = 0
    max_val = 0
    for algo_index, (algo, error) in enumerate(errors_dict.items()):
        line_style = 'solid'
        if both_oracle_and_blind_present:
            line_style = '-.' if (
                        'cmwf' in algo.lower() or 'prop' in algo.lower() or 'cmvdr' in algo.lower()) else 'dashed'
        if plot_diff_between_algos:
            ax.plot(error, linestyle=line_style)
        else:
            ax.plot(error, label=algo, linestyle=line_style)
        min_val = min(np.min(error), min_val)
        max_val = max(np.max(error), max_val)

    if which_axis == 'time':
        ax.set_xlabel('Time frame')
    else:
        ax.set_xlabel('Frequency [Hz]')
        # use dft_props['fs'] to convert the frequency bins to Hz
        freqs_hz = np.fft.rfftfreq(dft_props['nfft'], 1 / dft_props['fs'])
        freqs_hz = freqs_hz[:K_nfft_real]
        spacing_hz = 1000
        spacing = spacing_hz / (freqs_hz[1] - freqs_hz[0])
        list_ticks = np.arange(0, K_nfft_real, spacing)
        while list_ticks.size < 16:
            spacing = spacing // 2
            list_ticks = np.arange(0, K_nfft_real, spacing)
        ax.set_xticks(list_ticks)
        list_ticks = [int(x) for x in list_ticks]
        ax.set_xticklabels([f"{freqs_hz[kk]:.0f}" for kk in list_ticks], rotation=45, fontsize='medium')

    ax.set_ylabel('RMSE')
    ax.grid()
    ax.set_ylim([min(min_val, -0.11), max(max_val, 0.11)])
    if plot_diff_between_algos:
        ax.set_title(f'Difference in RMSE between cMVDR and MVDR (Lower than 0 -> cMVDR better)')
    else:
        ax.set_title(f'RMSE between {ref_clean} and other signals (over {which_axis})')
        ax.legend(fontsize='large')
    fig.show()

    return fig


def plot_waveforms_and_spectrograms(plot_settings, signals_dict, dft_props, alpha_list, num_chunks, slice_bf_list,
                                    K_nfft, delta_t, freq_max_cyclic=10000, debug_title='',
                                    benchmark_algo='mvdr_blind'):
    max_time_frames = np.iinfo(np.int32).max
    freqs_hz = np.fft.fftfreq(K_nfft, 1 / dft_props['fs'])

    # Override some fields
    plot_settings['destination'] = 'debug'
    plot_settings = config.ConfigManager.get_plot_settings(plot_settings)
    plot_settings['debug_title'] = debug_title

    if plot_settings['spectrograms']:
        print("Plotting spectrograms.")
        max_displayed_frequency_bin = K_nfft // 2 + 1
        # max_displayed_frequency_bin = K_nfft // 16
        debug_title = plot_settings.get('debug_title', '')
        plot_spectrograms(signals_dict, freqs_hz, max_displayed_frequency_bin=max_displayed_frequency_bin,
                          max_time_frames=max_time_frames, save_figs=plot_settings['save_plots'],
                          slice_chunks=slice_bf_list, delta_t=delta_t, suptitle=debug_title)

    if plot_settings['waveforms']:
        print("Plotting waveforms.")
        plot_waveforms(signals_dict, dft_props, alpha_list, num_chunks, slice_bf_list,
                       save_plots=plot_settings['save_plots'])

    if plot_settings['error_per_frequency']:
        # pl.plot_rmse_before_average(signals=signals_dict, K_nfft_real=K_nfft_real, which_axis='time')
        signals_dict_no_oracle = {key: val for key, val in signals_dict.items() if 'oracle' not in key}
        plot_rmse_before_average(signals=signals_dict_no_oracle, dft_props=dft_props, which_axis='freq',
                                 max_freq_hz=min(dft_props['fs'] / 2, freq_max_cyclic),
                                 plot_diff_between_algos=True, benchmark=benchmark_algo)

    return None


def plot_f0_spectrogram_outer(s, f0_over_time, dft_props, sig_name=''):
    R_shift_samples = dft_props['r_shift_samples']
    fs = dft_props['fs']
    nfft = dft_props['nfft']

    S_pow_stft = np.abs(librosa.stft(s, n_fft=nfft, hop_length=R_shift_samples)) ** 2
    S_pow_stft = S_pow_stft[..., :f0_over_time.shape[-1]]

    # Plot spectrogram and f0 superimposed
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    S_db = librosa.amplitude_to_db(S_pow_stft, ref=np.max)
    librosa.display.specshow(S_db,
                             sr=fs, hop_length=R_shift_samples, x_axis='time', y_axis='linear', ax=ax)
    ax.plot(np.arange(len(f0_over_time)) * R_shift_samples / fs, f0_over_time, color='g', label='f0', linewidth=1.5)
    ax.set_title(f'{sig_name.capitalize()} sound. Power spectrogram and f0.')

    ax.legend()

    plt.show()

    return fig, ax
