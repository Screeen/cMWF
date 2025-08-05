import numpy as np


class CovarianceHolder:
    def __init__(self,
                 noisy_nb=np.array([], dtype=complex, order='F'),
                 noise_nb=np.array([], dtype=complex, order='F'),
                 noisy_wb=np.array([], dtype=complex, order='F'),
                 noise_wb=np.array([], dtype=complex, order='F'),
                 wet_nb=np.array([], dtype=complex, order='F'),
                 wet_wb=np.array([], dtype=complex, order='F'),
                 cross_noisy_early_nb=np.array([], dtype=complex, order='F'),
                 cross_noisy_early_wb=np.array([], dtype=complex, order='F'),
                 noisy_pseudo=np.array([], dtype=complex, order='F'),
                 # cross_wet_early_wb=np.array([], dtype=complex, order='F'),
                 # var_early=np.array([], dtype=complex, order='F'),
                 # noise_wb_eval=np.array([], dtype=complex)
                 ):
        """
        Consider the model X = AS + V = D + V, where A is (relative) transfer function, S is target signal and N is noise
        """
        self.noisy_nb = noisy_nb  # E{xx^H}
        self.noise_nb = noise_nb  # E{vv^H}
        self.wet_nb = wet_nb  # E{dd^H}

        self.noisy_wb = noisy_wb
        self.noise_wb = noise_wb
        self.wet_wb = wet_wb

        self.cross_noisy_early_nb = cross_noisy_early_nb  # E{xd^H}
        self.cross_noisy_early_wb = cross_noisy_early_wb

        self.noisy_pseudo = noisy_pseudo  # E{xx^T}

        # self.cross_wet_early_wb = cross_wet_early_wb
        # self.var_early = var_early
        # self.noise_wb_eval = noise_wb_eval  # only needed for SINR calculation

    def is_empty(self):
        all_variables = [getattr(self, key) for key in self.__dict__.keys()]
        return all([var.size == 0 for var in all_variables])

    def get_as_dict(self):
        # Return all variables as a dictionary. Notice that changing the dictionary will not change the object.
        return {key: getattr(self, key) for key in self.__dict__.keys()}

    def same_size(self, ch_other):
        all_self_variables = [getattr(self, key) for key in self.__dict__.keys()]
        all_other_variables = [getattr(ch_other, key) for key in ch_other.__dict__.keys()]
        return all(
            [var_self.size == var_other.size for var_self, var_other in zip(all_self_variables, all_other_variables)])

    def has_compatible_size(self, sig_shape_k_m_p):
        # Check if the signal shape has changed (M or P). In that case, we need to reallocate.
        mp_unchanged = self.noisy_wb.shape[-1] == sig_shape_k_m_p[-1] * sig_shape_k_m_p[-2]
        k_unchanged = self.noisy_wb.shape[0] == sig_shape_k_m_p[0]
        return mp_unchanged and k_unchanged
