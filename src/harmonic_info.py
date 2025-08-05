import warnings

import numpy as np
from . import utils as u


class HarmonicInfo:
    """ Class to store information about the harmonic structure of the signal."""

    def __init__(self,
                 harmonic_bins=np.array([]),
                 harmonic_sets=np.array([]),
                 harmonic_freqs_hz=np.array([]),
                 alpha_mods_sets=None
                 ):
        """
        There are K frequency bins and P harmonic sets.

        :param harmonic_bins: The indices of the harmonic bins. E.g., [3, 4, 6, 7, 9, 12, 13, 14]
        :param harmonic_sets: Size K, same as DFT frequency bins. Harmonic sets by index.
                              Each bin is assigned to a set from 0 to P-1. value is -1 if the bin is not harmonic.
        :param harmonic_freqs_hz: Size P, list of harmonic frequencies. Each harmonic is the "center frequency" of a set.
        :param alpha_mods_sets: Size P, list of np.ndarray. Each array contains the modulations for the corresponding harmonic set.
        """

        self.harmonic_bins = harmonic_bins

        # Size K, same as DFT frequency bins. True if the bin is harmonic, False otherwise.
        self.mask_harmonic_bins = np.array([])

        self.harmonic_sets = harmonic_sets
        self.harmonic_sets_before_coh = np.array([])  # local coherence filtering only

        if harmonic_bins.size > 0 and harmonic_sets.size > 0:
            self.mask_harmonic_bins = self.get_mask_harmonic_bins(harmonic_sets.size, harmonic_bins)

        self.harmonic_freqs_hz = harmonic_freqs_hz

        self.alpha_mods_sets = [np.array([])]
        if alpha_mods_sets:
            self.alpha_mods_sets = alpha_mods_sets

        # Size P, number of shifts for each harmonic set. E.g., [3, 3, 2, 1] for a signal with 4 harmonic sets.
        self.num_shifts_per_set = np.array([])
        self.num_shifts_per_set_before_coh = np.array([])  # local coherence filtering only
        if alpha_mods_sets:
            # Count number of modulations in each harmonic set
            self.num_shifts_per_set = self.get_num_shifts_per_set(alpha_mods_sets)

    def get_harmonic_set_and_num_shifts(self, kk, before_coherence=False):
        """
        Get the harmonic set index and the number of shifts for a given frequency bin index.
        :param kk: Frequency bin index.
        :return: Harmonic set index and number of shifts.
        """

        harmonic_sets_ = self.harmonic_sets
        num_shifts_per_set_ = self.num_shifts_per_set
        if before_coherence:  # local coherence filtering only
            harmonic_sets_ = self.harmonic_sets_before_coh
            num_shifts_per_set_ = self.num_shifts_per_set_before_coh

        if harmonic_sets_.size == 0 or harmonic_sets_[kk] == -1:
            # By default, assign to harmonic_set 0 and do not shift
            harmonic_set_kk = 0
            P = 1
        else:
            # harmonic_set_kk is the harmonic set that corresponds to bin kk
            harmonic_set_kk = harmonic_sets_[kk]

            # P is the number of shifts associated with harmonic_set_kk
            P = num_shifts_per_set_[harmonic_set_kk]

        return harmonic_set_kk, P

    def get_num_shifts_all_frequencies(self):
        """ Get P for all frequency bins. """

        P_all = np.zeros(len(self.harmonic_sets), dtype=int)
        for kk in range(len(self.harmonic_sets)):
            _, P_all[kk] = self.get_harmonic_set_and_num_shifts(kk)

        return P_all

    @staticmethod
    def get_num_shifts_per_set(alpha_mods_sets):
        """
        Count the length of each array in alpha_mods_sets.
        Each array contains the modulations applied to some harmonic set.
        """
        return np.array([len(alpha_vec) for alpha_vec in alpha_mods_sets])

    @staticmethod
    def get_mask_harmonic_bins(K_nfft_real, harmonic_bins):
        mask_harmonic_bins = np.zeros((K_nfft_real,), dtype=bool)
        if harmonic_bins.size == 0:
            return mask_harmonic_bins
        mask_harmonic_bins[harmonic_bins] = True
        return mask_harmonic_bins
