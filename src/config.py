import copy
import warnings

import yaml
import logging
from pathlib import Path
import numpy as np
from src import globs as gs
import sys

# Logger
logger = logging.getLogger(__name__)
config_folder_source = Path(__file__).parent.parent / "configs"


class ConfigManager:
    def __init__(self):
        self.ir_paths = {}
        self.angle_idx = 0

    # @staticmethod
    # def get_impulse_response_paths_fixed():
    #     project_path = Path(__file__).resolve().parent.parent
    #     datasets_path = project_path.parent / 'datasets'
    #     ir_dataset_path = datasets_path / 'Hadad_shortcut'
    #     speech_ir_path = ir_dataset_path / '2.wav'
    #     dir_noise_ir_path = ir_dataset_path / '1.wav'
    #
    #     return speech_ir_path, dir_noise_ir_path

    def get_impulse_response_paths(self, target_angle=None, rir_specs=None):

        if rir_specs['dataset_name'].lower() == 'handpicked':
            common_path = Path(rir_specs['common_path'])
            assert common_path.exists() and common_path.is_dir()
            rirs = [rir_specs['target_path'], rir_specs['noise_path']]
            rirs = [common_path / rir_path for rir_path in rirs]
            assert all(p.exists() and p.is_file() for p in rirs)
            return rirs

        datasets_path = Path(__file__).resolve().parent.parent.parent / 'datasets'
        rt60, distance, mic_spacing = rir_specs['rt60'], rir_specs['distance'], rir_specs['mic_spacing']
        ir_folder = f'Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_{rt60}__{mic_spacing}'
        datasets_path = datasets_path / 'Hadad_wav' / ir_folder

        # Pick two random wav files from datasets_path. They have to be different. Use the following code to pick them:
        # Example name: Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.360s)_8-8-8-8-8-8-8_1m_015.wav
        if self.ir_paths.get(ir_folder) is None:
            if distance == "any":
                ir_files = list(datasets_path.glob(f'*.wav'))
            else:
                ir_files = list(datasets_path.glob(f'*_{distance}_*.wav'))
            ir_paths = [datasets_path / ir_file for ir_file in ir_files]
            self.ir_paths[ir_folder] = ir_paths
        else:
            ir_paths = self.ir_paths[ir_folder]

        if len(ir_paths) < 2:
            raise ValueError(f"Could not find two impulse response files in {datasets_path}.")

        if target_angle is None or target_angle == 'None':
            ret_paths = gs.rng.choice(ir_paths, 2, replace=False)
        else:
            # target angle is fixed, noise angle is random
            target_ir_path = [ir_path for ir_path in ir_paths if target_angle in ir_path.name]
            noise_ir_path = [ir_path for ir_path in ir_paths if rir_specs['noise_angle'] in ir_path.name]
            if len(target_ir_path) == 0:
                raise ValueError(f"Could not find angle {target_angle} in the file names.")
            ret_paths = [target_ir_path[0], noise_ir_path[0]]

        # print(f"{ret_paths[0].name} \n{ret_paths[1].name}")
        return ret_paths

    @staticmethod
    def check_varying_params_have_default_values(varying_params, default_values):
        for param_name, param_values in varying_params.items():
            if param_name not in default_values.keys():
                raise ValueError(f"Parameter '{param_name}' has no default value.")

    @classmethod
    def get_parameters_and_default_values(cls, parameters, parameter_to_vary='', default_values=None):
        # Define your parameters and their values

        if parameter_to_vary in parameters:
            varying_param_values = parameters[parameter_to_vary]
        elif parameter_to_vary != '':
            raise ValueError(f"Parameter {parameter_to_vary} not found in parameters.")
        else:
            varying_param_values = [None]

        ConfigManager.check_varying_params_have_default_values(parameters, default_values)

        return parameters, default_values, varying_param_values

    @staticmethod
    def get_plot_settings(plot_settings_in):

        plot_settings_out = plot_settings_in.copy()

        # plot_destination: debug, slides, paper
        for_export = plot_settings_in['destination'] != 'debug'

        plot_settings_out.update({
            'show_date_plots': False if for_export else True,
            'use_tex': True if for_export else False,
            'save_plots': True if for_export else False,
            'separately': True if plot_settings_in['destination'] == 'paper' else False,
            'force_no_plots': False,
            'show_title': True,
            'show_legend': True,
        })

        # Enable/disable plots
        plot_types = ['f0_spectrogram', 'spectrograms', 'all_variations', 'waveforms', 'error_per_frequency']
        for plot_type in plot_types:
            plot_settings_out[plot_type] = plot_settings_in.get(plot_type, False)  # True: plot, False: don't plot
            if plot_settings_out['force_no_plots']:
                plot_settings_out[plot_type] = False

        return plot_settings_out

    @staticmethod
    def is_speech(signal_cfg):

        if signal_cfg['sig_type'] != 'sample':
            return False

        target_path_str = str(signal_cfg['sample_name']).lower()
        voice_samples = ['male', 'female', '430a010a', 'long_a', 'ka', '432a010a', 'harvard']

        for voice_sample in voice_samples:
            if voice_sample in target_path_str:
                return True

        return False

    @staticmethod
    def choose_signals_to_modulate(cyclostationary_target_, minimize_noisy_cov_mvdr_, is_first_chunk,
                                   recursive_averaging, name_input_sig):
        """ Only modulated signals that are actually needed to save computations """

        signals_to_modulate = [name_input_sig]

        if cyclostationary_target_:  # for cMWF
            signals_to_modulate.append('wet_rank1')
            signals_to_modulate.append('noise_cov_est')

        # print("DEBUG")  # always add noise_cov_est as it may be used for coherence?
        if 'noise_cov_est' not in signals_to_modulate and (not minimize_noisy_cov_mvdr_):
            signals_to_modulate.append('noise_cov_est')  # for cMVDR when minimizing the noise covariance

        # Modulate noise at first chunk when doing recursive avg so that we can initialize noisy_wb to noise_wb
        if 'noise_cov_est' not in signals_to_modulate and is_first_chunk and recursive_averaging:
            signals_to_modulate.append('noise_cov_est')

        return signals_to_modulate
    
    @staticmethod
    def get_varying_parameters_names(cfg_default):
        
        varying_parameters_names = cfg_default['varying_parameters_names']
        varying_parameters_names = varying_parameters_names[:cfg_default['max_num_varying_parameters']]
        return varying_parameters_names


def get_varying_param_values(configuration: dict, parameter_to_vary: str):  # -> List[Union[str, int, float]]:
    """ Read varying parameter values from configuration file """

    # to_convert_to_float = ['f0_hz', 'cov_estimation|loading', 'noise|inharmonicity_percentage']
    to_convert_to_float = ['cov_estimation|loading', 'noise|inharmonicity_percentage',
                           'target|inharmonicity_percentage']

    if parameter_to_vary in configuration['varying_params_values']:
        varying_param_values = configuration['varying_params_values'][parameter_to_vary]
        varying_param_values = varying_param_values[:configuration['max_num_variations_per_parameter']]
        if parameter_to_vary in to_convert_to_float:
            varying_param_values = [float(val) for val in varying_param_values]
    else:
        raise ValueError(f"{parameter_to_vary = } not found in parameters."
                         f"Available parameters: {list(configuration['varying_params_values'].keys())}")

    return varying_param_values


def load_configuration(cfg_name=None):
    """ Load configuration file """

    print(f"Loading configuration file: {cfg_name}")
    cfg_path = config_folder_source / cfg_name
    cfg_custom = load_configuration_from_path(cfg_path)
    return cfg_custom


def load_configuration_from_path(configuration_path):
    """ Read settings from configuration file """

    with open(configuration_path, 'r') as f:
        conf_dict = yaml.safe_load(f)
    logger.info(f"Loaded configuration file {configuration_path}")
    return conf_dict


def write_configuration(config_dict, dest_path):
    """ Write configuration file """

    # dest_path = config_folder_source / config_name
    with open(dest_path, 'w') as outfile:
        yaml.dump(config_dict, outfile)


def set_stft_properties(stft_props, fs):
    """ Set STFT properties like window length, overlap, etc. based on the configuration values """

    stft_props['nw'] = stft_props['nfft']
    stft_props['noverlap'] = int(stft_props['nw'] * stft_props['overlap_fraction'])
    stft_props['r_shift_samples'] = stft_props['nw'] - stft_props['noverlap']
    stft_props['nfft_real'] = stft_props['nfft'] // 2 + 1
    stft_props['fs'] = fs

    return stft_props


def get_config_single_experiment(cfg_default, exp_name):
    """ Get configuration for a single experiment (other names, such as 'M', are not included) """

    cfg_single_exp = cfg_default.copy()
    cfg_single_exp['varying_parameters_names'] = exp_name

    return cfg_single_exp


def get_config_single_variation(cfg_exp, idx_var, varying_parameter_name):
    """ Get configuration for a single variation (e.g. SNR = 0 dB) """

    cfg_single_var = copy.deepcopy(cfg_exp)

    # means that the parameter is within a dictionary, for example noise: snr_db_dir = 0
    if '|' in varying_parameter_name:
        split_name = varying_parameter_name.split('|')  # could be 2 or more elements
        if len(split_name) == 2:
            dict_name, param_name = split_name
            cfg_single_var[dict_name][param_name] = cfg_exp['varying_params_values'][varying_parameter_name][idx_var]
        elif len(split_name) == 3:
            dict_name, sub_dict_name, param_name = split_name
            cfg_single_var[dict_name][sub_dict_name][param_name] = cfg_exp['varying_params_values'][varying_parameter_name][idx_var]
        else:
            raise NotImplementedError
    else:
        cfg_single_var[varying_parameter_name] = cfg_exp['varying_params_values'][varying_parameter_name][idx_var]

    return cfg_single_var


def check_cyclic_target_or_not(cfg):

    try:
        cyclostationary_target = cfg['cyclostationary_target']
        sig_type = cfg['target']['sig_type'],
        noise_type = cfg['noise']['sig_type']
        if cyclostationary_target and 'white' in sig_type:
            raise ValueError(
                f"{cyclostationary_target = } (implying that target is cyclic) but {sig_type = }!")
        elif not cyclostationary_target and 'white' in noise_type:
            warnings.warn(f"{cyclostationary_target = } (implying that noise is cyclic) but {noise_type = }!")
    except KeyError as e:
        print(f"{e} in check_cyclic_target_or_not")


def update_target_settings(target_sett):
    if target_sett['sig_type'] != 'sinusoidal' and target_sett['sig_type'] != 'sinusoidal_varying_freq':
        target_sett['f0_hz'] = [np.nan, np.nan]
        target_sett['num_harmonics'] = np.nan
    return target_sett


def merge_configurations(cfg_primary, cfg_secondary):
    # We can't simply do cfg_default.update(cfg_secondary) because we need to merge dictionaries within dictionaries
    # It could be that cfg_secondary only specifies a few keys of inner dictionaries.
    # Do it recursively
    for key, value in cfg_secondary.items():
        if isinstance(value, dict):
            if key not in cfg_primary:
                cfg_primary[key] = value
            else:
                merge_configurations(cfg_primary[key], value)
        else:
            cfg_primary[key] = value
    return cfg_primary


def adjust_config_for_debug(cfg_default):
    if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
        cfg_default['num_montecarlo_simulations'] = 1
        cfg_default['plot']['destination'] = 'debug'
        print(f"Running in debug mode. {cfg_default['num_montecarlo_simulations'] = } and "
                f"{cfg_default['plot']['destination'] = }")
    return cfg_default


def load_and_merge_secondary_config(cfg_default):
    if cfg_default['secondary_config_file'].lower() != 'none':
        cfg_secondary = load_configuration(cfg_default['secondary_config_file'])
        data_type_selected = cfg_default['data_type']
        cfg_secondary_data_type = cfg_secondary.get(data_type_selected, {})

        # Remove the data_type key from the secondary config, so that it doesn't overwrite the default config
        # when merging the two configurations
        cfg_secondary.pop(data_type_selected)

        cfg_default = merge_configurations(cfg_default, cfg_secondary)
        cfg_default = merge_configurations(cfg_default, cfg_secondary_data_type)

        # Remove the unused "data types" (configuration settings) to avoid polluting the final YAML file stored with the experiment
        for data_type_other in cfg_default['data_types_all']:
            if data_type_other != data_type_selected:
                cfg_default.pop(data_type_other, None)

        cfg_default = adjust_config_for_debug(cfg_default)
    return cfg_default


def assign_default_values(cfg):
    if 'minimize_noisy_cov_mvdr' not in cfg['beamforming']:
        cfg['beamforming']['minimize_noisy_cov_mvdr'] = True

    cfg['harmonics_est']['freq_range_cyclic'] = cfg['cyclic']['freq_range_cyclic']

    cfg['target']['harmonic_correlation'] = cfg['target'].get('harmonic_correlation', 1.)
    cfg['noise']['harmonic_correlation'] = cfg['noise'].get('harmonic_correlation', 1.)
    cfg['noise']['skip_convolution_rir'] = cfg['noise'].get('skip_convolution_rir', False)
    cfg['noise']['noise_var_rtf'] = cfg['noise'].get('noise_var_rtf', 0)

    if 'chunk_len_seconds' not in cfg['time']:
        raise AttributeError("chunk_len_seconds must be specified in YAML file in the 'time' dict.")
    cfg['time']['duration_approx_seconds'] = cfg['time'].get('duration_approx_seconds', cfg['time']['chunk_len_seconds'])
    cfg['time']['fixed_length_chunks'] = cfg['time'].get('fixed_length_chunks', True)
    cfg['time']['overlap_frame_cov_est_percentage'] = cfg['time'].get('overlap_frame_cov_est_percentage', 0)

    cfg['cyclic']['use_global_coherence'] = cfg['cyclic'].get('use_global_coherence', False)
    cfg['cyclic']['harmonic_threshold'] = cfg['cyclic'].get('harmonic_threshold', 0)

    cfg['use_masked_stft_for_evaluation'] = cfg.get('use_masked_stft_for_evaluation', False)
    cfg['print_results'] = cfg.get('print_results', True)
    cfg['rir_specs']['use_real_rirs'] = cfg['rir_specs'].get('use_real_rirs', True)

    cfg['metrics']['freq'] = cfg['metrics'].get('freq', [])
    cfg['metrics']['other'] = cfg['metrics'].get('other', [])

    cfg['cov_estimation']['use_rank1_model_for_oracle_cov_wet_estimation'] = cfg['cov_estimation'].get(
        'use_rank1_model_for_oracle_cov_wet_estimation', True)

    return cfg
