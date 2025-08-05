import numpy as np
from pathlib import Path
from datetime import datetime
from . import utils as u
import matplotlib
from matplotlib.ticker import MaxNLocator

from .plotter import get_display_name


class Player:

    def __init__(self):
        pass

    @staticmethod
    def get_dir_path():
        dir_path = Path(f'demos') / Path(f"{datetime.today().strftime('%Y-%m-%d')}")
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path

    @staticmethod
    def plot_mel_spectrograms(signals, for_spectrogram, fs=16000, save_figs=False):

        import librosa
        import matplotlib.pyplot as plt
        signals_spec_db = {}

        dir_path = Player.get_dir_path()

        # Step 1: Compute mel spectrograms and convert to dB using constant ref
        for key in for_spectrogram:
            mel = librosa.feature.melspectrogram(y=signals[key], sr=fs, n_mels=512, fmax=fs // 2)
            S_dB = librosa.power_to_db(mel, ref=1000.0)
            signals_spec_db[key] = S_dB

        # Step 2: Global min and max for consistent color range
        min_val = min(np.min(S) for S in signals_spec_db.values())
        max_val = max(np.max(S) for S in signals_spec_db.values())

        # Step 3: Plot with fixed color scale
        for name, S_dB in signals_spec_db.items():
            fig, ax = plt.subplots(1, 1)
            pcm = librosa.display.specshow(S_dB, sr=fs, x_axis='time', y_axis='mel', fmax=8000,
                                     vmin=min_val, vmax=max_val, cmap='magma', ax=ax)
            fig.colorbar(format='%+2.0f dB', mappable=pcm)
            ax.set_title(name)
            fig.tight_layout()
            fig.show()

            if save_figs:
                u.savefig(fig, dir_path / f'{name}.png')

    @staticmethod
    def extract_numeric_label(label):
        # Remove LaTeX wrappers like '$\\mathdefault{512}$' → '512'
        import re
        text = label.get_text()
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
        return float(match.group()) if match else None

    @staticmethod
    def encode_label(value, pos):
        # Create a matplotlib.text.Text object like Text(x=pos, y=value, text='$\\mathdefault{512}$')
        if value.is_integer():
            val_str = str(int(value))
        else:
            val_str = f"{value:.1f}"
        math_str = f"$\\mathdefault{{{val_str}}}$"
        return matplotlib.text.Text(x=0, y=pos, text=math_str)

    @staticmethod
    def plot_mel_spectrograms_2_by_2(signals, for_spectrogram, fs=16000, save_figs=False):
        """ Plot mel spectrograms in a 2x2 grid. """

        import librosa
        import matplotlib.pyplot as plt
        signals_spec_db = {}

        # Step 1: Compute mel spectrograms and convert to dB using constant ref
        for key in for_spectrogram:
            mel = librosa.feature.melspectrogram(y=signals[key], sr=fs, n_mels=512, fmax=fs // 2)
            S_dB = librosa.power_to_db(mel, ref=1000.0)
            signals_spec_db[key] = S_dB

        # Step 2: Global min and max for consistent color range
        min_val = min(np.min(S) for S in signals_spec_db.values())
        max_val = max(np.max(S) for S in signals_spec_db.values())

        fig_size = u.get_plot_width_double_column_latex(), u.get_plot_width_double_column_latex() * 0.9
        fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=fig_size,
                               sharex=True, sharey=True)

        # Step 3: Plot with fixed color scale
        pcm = None
        for i, (name, S_dB) in enumerate(signals_spec_db.items()):
            row, col = divmod(i, 2)
            pcm = librosa.display.specshow(S_dB, sr=fs, x_axis='time', y_axis='mel', fmax=8000,
                                           vmin=min_val, vmax=max_val, cmap='magma', ax=axs[row, col])
            name_disp = get_display_name(name)
            axs[row, col].set_title(name_disp)

            # Get current tick positions and labels
            tick_locs = axs[row, col].get_yticks()
            tick_labels = [Player.encode_label(val / 1000, pos) for val, pos in zip(tick_locs, tick_locs)]
            axs[row, col].set_yticks(tick_locs, labels=tick_labels)
            axs[row, col].tick_params(axis='x', pad=1)  # smaller pad → labels closer to ticks
            axs[row, col].tick_params(axis='y', pad=1)  # smaller pad → labels closer to ticks
            axs[row, col].set_ylabel("Frequency [kHz]")
            axs[row, col].set_xlabel("Time [s]")

        # Colorbar is shared across all subplots (to the right of the 2x2 grid, as tall as 2 rows)
        cbar = fig.colorbar(format='%+2.0fdB', mappable=pcm, ax=axs, orientation='vertical', fraction=0.06, pad=0.04,
                     aspect=30)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # maximum 4 ticks
        cbar.ax.tick_params(axis='y', pad=1)  # smaller pad brings labels closer to the bar

        # To avoid repeating xlabels and ylabels in each subplot
        for a in axs.flat:
            a.label_outer()

        fig.show()

        dir_path = Player.get_dir_path()
        name = '_'.join(for_spectrogram)
        if save_figs:
            u.savefig(fig, dir_path / f'{name}.pdf')

    @staticmethod
    def save_wavs(signals, fs):

        dir_path = Player.get_dir_path()

        u.save_wav(signals['noisy'], dir_path / 'noisy.wav', fs)
        u.save_wav(signals['mvdr_blind'], dir_path / 'mvdr_blind.wav', fs)
        u.save_wav(signals['cmvdr_blind'], dir_path / 'cmvdr_blind.wav', fs)

        u.save_wav(signals['noise'], dir_path / 'noise.wav', fs)
        u.save_wav(signals['wet'], dir_path / 'wet.wav', fs)
        u.save_wav(signals['early'], dir_path / 'early.wav', fs)

        u.save_wav(signals['mwf_blind'], dir_path / 'mwf_blind.wav', fs)
        u.save_wav(signals['cmwf_blind'], dir_path / 'cmwf_blind.wav', fs)

        u.save_wav(signals['clcmv_blind'], dir_path / 'clcmv_blind.wav', fs)

        u.save_wav(signals['mvdr_semi-oracle'], dir_path / 'mvdr_known_rtf.wav', fs)
        u.save_wav(signals['cmvdr_semi-oracle'], dir_path / 'cmvdr_known_rtf.wav', fs)

    @staticmethod
    def play_signals(signals_dict_all_variations_time, fs):
        """ Play the signals. """
        signals = signals_dict_all_variations_time[list(signals_dict_all_variations_time.keys())[1]][0]
        sett = {'fs': fs, 'max_length_seconds': min(10, int(len(signals['noisy']) / fs)), 'volume': 0.3, 'smoothen_corners_flag': True}

        if 0:
            u.play(signals['wet'], **sett)
            u.play(signals['noise'], **sett)
            u.play(signals['noise_cov_est'], **sett)
            u.play(signals['noisy'], **sett)

            #### MVDR
            u.play(signals['mvdr_blind'], **sett)
            u.play(signals['cmvdr_blind'], **sett)
            u.play(signals['cmvdr-wl_blind'], **sett)

            u.play(signals['mvdr_semi-oracle'], **sett)
            u.play(signals['cmvdr_semi-oracle'], **sett)

            #### MWF
            u.play(signals['mwf_blind'], **sett)
            u.play(signals['cmwf_blind'], **sett)

            u.play(signals['mwf_semi-oracle'], **sett)
            u.play(signals['cmwf_semi-oracle'], **sett)

            u.play(signals['mwf_oracle'], **sett)
            u.play(signals['cmwf_oracle'], **sett)

        if 0:
            Player.save_wavs(signals, fs)

        if 0:
            # for_spectrogram = ['mvdr_semi-oracle', 'cmvdr_semi-oracle', 'noisy']
            # for_spectrogram = ['noisy', 'wet', 'mwf_blind', 'cmwf_blind']
            # for_spectrogram = ['mwf_blind', 'cmwf_blind']
            for_spectrogram = ['noisy', 'wet_rank1', 'mvdr_blind', 'cmvdr_blind']
            # for_spectrogram = ['mvdr_blind', 'cmvdr_blind']
            Player.plot_mel_spectrograms(signals, for_spectrogram, fs=fs, save_figs=False)
            from src import player
            player.Player.plot_mel_spectrograms_2_by_2(signals, for_spectrogram, fs=fs, save_figs=False)
            # player.Player.plot_mel_spectrograms_2_by_2(signals, for_spectrogram, fs=fs, save_figs=True)
            # Player.plot_mel_spectrograms(signals, for_spectrogram, fs=fs, save_figs=True)
