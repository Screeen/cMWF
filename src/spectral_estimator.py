import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy


class SpectralEstimator:
    """ Estimate harmonic frequencies of a signal using the periodogram or Welch estimators. """

    def __init__(self):
        pass

    @staticmethod
    def estimate_harmonics_periodogram(waveform_cyclo, cfg_harm, fs, do_plots=False):
        """
        Estimate peaks in the spectrum using the periodogram. If signal is harmonic, peaks correspond to harmonic frequencies.
        """
        f_range_hz = cfg_harm['freq_range_cyclic']
        wait_len_seconds = cfg_harm['wait_len_seconds']
        # spectral_estimators = ['welch', 'periodogram', ]
        spectral_estimators = ['periodogram', ]

        # Wait up to 'wait_len' before estimating harmonics as there might be a transient at the beginning.
        # If signal is too short, wait as much as possible but use as many samples as requested.
        wait_len = int(wait_len_seconds * fs)
        target_len = int(min(waveform_cyclo.size, cfg_harm['max_len_seconds'] * fs))
        end_idx = min(waveform_cyclo.size, wait_len + target_len)
        start_idx = max(0, end_idx - target_len)
        x = waveform_cyclo[start_idx:end_idx]

        # y = {'psd': {}, 'peaks_hz': {}, 'peaks_hz_sorted': {} }
        y = {key: {} for key in spectral_estimators}
        for spec_est in spectral_estimators:
            if 'periodogram' in spec_est:
                f = np.fft.rfftfreq(cfg_harm['nfft'], 1 / fs)
                x_est = (np.abs(np.fft.rfft(x, n=cfg_harm['nfft'])) ** 2) / cfg_harm['nfft']
            elif 'welch' in spec_est:
                f, x_est = sp.signal.welch(x, fs, window='rect', nfft=cfg_harm['nfft'], nperseg=cfg_harm['nfft'] // 16,
                                           scaling='spectrum')
            else:
                raise ValueError

            # Only keep data within limits specified by f_range_hz
            res_hz = f[1] - f[0]
            f_range_bins = (int(f_range_hz[0] / res_hz), int(f_range_hz[1] / res_hz) + 1)
            f_hz = f[f_range_bins[0]:f_range_bins[1]]
            x_est = x_est[f_range_bins[0]:f_range_bins[1]]
            x_est = x_est / np.max(x_est)

            peaks = SpectralEstimator.find_spectral_peaks(x_est, f, cfg_harm)
            peaks_hz = f_hz[peaks]
            peaks_hz_sorted = np.sort(peaks_hz)
            # print(f"{spec_est = } {len(peaks_hz_sorted) = }, {peaks_hz_sorted}")

            y[spec_est]['psd'] = dcopy(x_est)
            y[spec_est]['peaks'] = dcopy(peaks)
            y[spec_est]['peaks_hz'] = dcopy(peaks_hz)
            y[spec_est]['peaks_hz_sorted'] = dcopy(peaks_hz_sorted)
            y[spec_est]['f_hz'] = dcopy(f_hz)

        if do_plots:
            fig, axs = plt.subplots(nrows=len(spectral_estimators), constrained_layout=True, sharex=False)
            axs = np.atleast_1d(axs)
            for (spec_est, ax) in zip(spectral_estimators, axs):
                yy = y[spec_est]
                ax.semilogy(yy['f_hz'], yy['psd'])
                ax.semilogy(yy['peaks_hz'], yy['psd'][yy['peaks']], "x", markersize=5, color='r', markeredgewidth=2)
                ax.set_xlim([0, f_range_hz[1]])  # show up to 500 Hz
                ax.set_ylim([1.e-8, 3.13])
                ax.set_xlabel("Frequency [Hz]")
                ax.set_title(f"{spec_est}")
                ax.grid()
            fig.show()

        selected_method = 'welch'
        selected_method = 'periodogram'

        return y[selected_method]['peaks_hz_sorted'], range(start_idx, end_idx)

    @staticmethod
    def find_spectral_peaks(x_est, f, cfg_harm):

        res_hz = f[1] - f[0]
        min_height = np.max(x_est) / cfg_harm['max_ratio_from_highest_peak']
        min_distance = max(1, cfg_harm['min_dist_hz'] / res_hz)
        peaks, peaks_infos = sp.signal.find_peaks(x_est, distance=min_distance, height=min_height,
                                                  width=cfg_harm['min_width'], prominence=cfg_harm['min_prominence'])

        # Retain up to max_num_peaks peaks
        peaks = peaks[np.argsort(peaks_infos['peak_heights'])[-cfg_harm['max_num_harmonics_peaks']:]]

        return peaks
