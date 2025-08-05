import copy

import numpy as np
import time
from copy import deepcopy as dcopy
from pathlib import Path
from datetime import datetime

from src import globs as gs, config

cfg_original = config.load_configuration('default.yaml')
cfg_original = config.load_and_merge_secondary_config(cfg_original)
gs.rng, cfg_original['seed_extracted'] = gs.compute_rng(cfg_original['seed_is_random'],
                                                        cfg_original['seed_if_not_random'])

from src.beamformer_manager import BeamformerManager
from src import (utils as u, data_generator, evaluator, plotter as pl, f0_manager)
from src.player import Player  # do not remove, useful for quick evaluation of signals from cmd line
from src.experiment_manager import ExperimentManager

u.set_printoptions_numpy()
threshold_hz_f0_std = np.inf


if __name__ == '__main__':

    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    results_data_type_plots = {}
    results_data_type_freq_est_plots = {}
    varying_parameters_names = config.ConfigManager.get_varying_parameters_names(cfg_original)
    plot_sett = config.ConfigManager.get_plot_settings(cfg_original['plot'])
    u.set_plot_options(use_tex=plot_sett['use_tex'])

    # quick check to make sure all parameters are valid to avoid errors later
    [config.get_varying_param_values(dcopy(cfg_original), x) for x in varying_parameters_names]

    for parameter_to_vary in varying_parameters_names:
        # parameter_to_vary could be 'P_max', 'f0_err_percent', SNR', etc.
        alpha_hz_printed_already = False
        weights_previous = {}

        cfg_default = dcopy(cfg_original)
        varying_param_values = config.get_varying_param_values(cfg_default, parameter_to_vary)
        fs = cfg_default['fs']
        cfg_default['target'] = config.update_target_settings(cfg_default['target'])
        config.check_cyclic_target_or_not(cfg_default)
        cfg_default['beamforming']['methods'] = BeamformerManager.infer_beamforming_methods(cfg_default['beamforming'])

        results_dict = {str(varying_param_value): [] for varying_param_value in varying_param_values}
        results_freq_est = dcopy(results_dict)
        signals_dict_all_variations_time = {str(param_val): {} for param_val in varying_param_values}

        anechoic, C_rtf = np.array([]), np.array([])
        signals_bfd_dict_backup = {}
        post_filtering = False

        # Each iteration is a different value of the parameter to vary. E.g., SNR = 0dB, then 10dB, etc.
        for idx_var, param_value in enumerate(varying_param_values):
            print(f"Varying parameter {parameter_to_vary}. Current value: {parameter_to_vary} = {str(param_value)}")
            cfg = config.get_config_single_variation(cfg_default, idx_var, parameter_to_vary)
            cfg = config.assign_default_values(cfg)
            dft_props = config.set_stft_properties(cfg['stft'], cfg['fs'])
            is_cmwf_bf = cfg['cyclostationary_target']

            # Each iteration is a different montecarlo realization. Randomly vary: target signal, noise, ATF, etc.
            for idx_mtc in range(cfg['num_montecarlo_simulations']):
                dg = data_generator.DataGenerator(cfg['target']['harmonic_correlation'],
                                                  cfg['noise']['harmonic_correlation'],
                                                  mean_random_proc=0.5 if is_cmwf_bf else 0.)
                f0man = f0_manager.F0Manager()

                if dg.sin_gen['target'].mean_random_process == 0 and is_cmwf_bf and \
                        cfg['target']['sig_type'] == 'sinusoidal':
                    raise ValueError("Only tested with cMVDR, could have unintended consequences with cMWF")

                do_plots = idx_mtc == 0 and (idx_var == 0 or idx_var == len(varying_param_values) - 1 or
                                                plot_sett['all_variations'])
                SFT, SFT_real, freqs_hz = dg.get_stft_objects(dft_props)
                signals, max_len_ir_atf, target_ir = dg.generate_signals(cfg, SFT_real, dft_props)

                harmonic_freqs_est, crb_dict, f0_over_time = f0man.estimate_f0_or_resonant_freqs(
                    signals, cfg, dft_props, sin_generators=dg.sin_gen,
                    do_plots=do_plots and cfg['plot']['f0_spectrogram'])

                # Covariance estimation & beamforming
                bfd_all_chunks_stft = ExperimentManager.run_cov_estimation_beamforming(
                    signals, f0man, f0_over_time, harmonic_freqs_est, cfg, dft_props, do_plots, SFT)

                signals_bfd_dict = ExperimentManager.convert_signals_time_domain(bfd_all_chunks_stft, SFT_real)

                signals_dict = {**signals, **signals_bfd_dict_backup, **signals_bfd_dict}
                signals_dict = evaluator.bake_dict_for_evaluation(signals_dict,
                                                                  needs_masked_stft=cfg['use_masked_stft_for_evaluation'])

                # Evaluate performance of beamformers. Single montecarlo and single parameter value (e.g., SNR = 0dB)
                metrics_list_time = evaluator.adjust_metrics_list(cfg['metrics']['time'],
                                                                  config.ConfigManager.is_speech(cfg['target']))
                results_dict[str(param_value)].append((
                    evaluator.evaluate_signals(signals_dict, metrics_list_time, cfg['metrics']['freq'], fs,
                                               cfg['use_masked_stft_for_evaluation'], reference_sig_name='wet_rank1',
                                               K_nfft_real=dft_props['nfft_real'], print_results=cfg['print_results'])))

                # Store audio signals for all param_values and montecarlo realizations to listen to them later
                signals_dict_all_variations_time[str(param_value)][idx_mtc] = {key: dcopy(signals_dict[key]['time']) for
                                                                               key in
                                                                               signals_dict.keys()}

                if 'oracle' not in cfg['harmonics_est']['algo'] and cfg['metrics']['other']:
                    harmonic_freqs_oracle = dg.sin_gen['noise'].freqs_synthetic_signal
                    res1 = evaluator.evaluate_frequency_estimation(harmonic_freqs_oracle, harmonic_freqs_est)
                    res2 = evaluator.evaluate_crb(crb_dict, harmonic_freqs_oracle, fs)
                    results_freq_est[str(param_value)].append({'freq-err-mae': res1 | res2})

            # end of montecarlo simulations loop
        # end of parameter variations loop

        plots_args_default = {'varying_param_values': varying_param_values, 'parameter_to_vary': parameter_to_vary}
        plot_args_bf, plot_args_freq_est = evaluator.rearrange_and_average_results_all(results_dict, plots_args_default,
                                                                                       results_freq_est, 'Noisy')
        evaluator.log_intermediate_results(plot_args_bf, plot_args_freq_est, varying_param_values, plot_sett)
        results_data_type_plots[parameter_to_vary] = dcopy(plot_args_bf)
        results_data_type_freq_est_plots[parameter_to_vary] = dcopy(plot_args_freq_est)

        try:
            Player.play_signals(signals_dict_all_variations_time, fs)
        except Exception as exc:
            print(f"Could not play the signals: {exc}")

        print(f"Varying parameter: {parameter_to_vary}")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f}s")

    target_path_figs = Path('figs') / datetime.now().strftime("%Y-%m-%d") / time.strftime('%Hh%M')

    # Plot beamforming errors
    pl.visualize_all_results(results_data_type_plots, plot_sett, cfg_original, False, False,
                             target_path_figs, True)

    # Plot errors for frequency estimation
    pl.visualize_all_results(results_data_type_freq_est_plots, plot_sett, cfg_original, plot_db=False,
                             target_path_figs_=target_path_figs)
    pl.visualize_all_results(results_data_type_freq_est_plots, plot_sett, cfg_original, plot_db=True,
                             target_path_figs_=target_path_figs)

    # Move *.pkl files from target_path_figs to target_path_figs / 'figs_pkl'
    if target_path_figs.exists() and any(target_path_figs.iterdir()):
        target_path_figs_pkl = target_path_figs / 'figs_pkl'
        target_path_figs_pkl.mkdir(parents=True, exist_ok=True)
        for pkl_file in target_path_figs.glob('*.pkl'):
            pkl_file.rename(target_path_figs_pkl / pkl_file.name)

    # Open the folder with the figures if the elapsed time is more than 60 seconds or if there are many simulations
    if elapsed_time > 60 or cfg_original['num_montecarlo_simulations'] > 10:
        import subprocess
        subprocess.run(["open", target_path_figs])
