import numpy as np
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import cdist

class BC:
    """
    This class calculates the mean Bray Curtis distance vector for a cohort or a single sample
    from a reference cohort.
    """
    def __init__(self, ref_cohort, cohort, cohort_ind_dict=None,
                 median=False, identical=False):
        """
        :param ref_cohort: a matrix, samples are in the rows.
        :param cohort: a matrix or vector, if matrix, samples are in the rows.
        :param second_cohort_ind_dict: if cohort is a matrix,a dictionary, keys are indices of
         samples in cohort, values are indices of samples in ref_cohort. If cohort is a vector, a tuple,
         contain indexes of ref_cohort.
        :param median: if True, the median distance is calculated, otherwise the
         mean distance is calculated.
         :return: If cohort is a matrix, mean or median distance vector. If cohort is a vector,
          mean or median distance value.
        """
        self.ref_cohort = ref_cohort
        self.cohort = cohort
        if self.ref_cohort.ndim != 2:
            raise ValueError("ref_cohort should be a 2D numpy array")
        if self.cohort.ndim not in [1, 2]:
            raise ValueError("cohort should be a 1D or 2D numpy array")
        self.ref_cohort = self._normalize_cohort(ref_cohort)
        self.cohort = self._normalize_cohort(cohort)
        if self.cohort.ndim == 1:
            self.num_samples_cohort = 1
            self.num_species = np.size(cohort)
        else:
            self.num_samples_cohort = cohort.shape[0]
            self.num_species = cohort.shape[1]
        self.num_samples_ref_cohort = ref_cohort.shape[0]
        self.cohort_ind_dict = cohort_ind_dict
        if self.cohort_ind_dict:
            print('a')
            if self.num_samples_cohort == 1:
                if not isinstance(self.cohort_ind_dict, tuple):
                    raise ValueError("cohort_ind_dict should be a tuple if cohort is a vector")
            if self.num_samples_cohort > 1:
                if not isinstance(self.cohort_ind_dict, dict):
                    raise ValueError("cohort_ind_dict should be a dictionary if cohort is a matrix")
        self.identical = identical
        self.median = median
        try:
            assert isinstance(self.identical, bool)
            assert isinstance(self.median, bool)
        except AssertionError:
            raise AssertionError("Invalid input values for identical or pos. They should be boolean values.")

    def _BC_sample_vs_cohort(self):
        if self.cohort_ind_dict:
            k = self.cohort_ind_dict[0]
            sample_dist = np.array([braycurtis(self.ref_cohort[j, :], self.cohort) for j in range(
                0, self.num_samples_ref_cohort) if j != k])
        else:
            sample_dist = np.array([braycurtis(self.ref_cohort[j, :], self.cohort) for j in range(
                0, self.num_samples_ref_cohort)])
        if self.median:  # measure median distance
            mean_dist_vector = np.median(sample_dist)
        else:  # measure mean distance
            mean_dist_vector = np.mean(sample_dist)
        return mean_dist_vector

    def _BC_cohort_vs_cohort(self):
        if self.identical:
            dist_matrix = cdist(self.cohort, self.ref_cohort, metric='braycurtis')
            dist_matrix_float = dist_matrix.astype(float)
            np.fill_diagonal(dist_matrix_float, np.nan)
            if self.median:  # measure median distance
                mean_dist_vector = np.nanmedian(dist_matrix_float, axis=1)
            else:  # measure mean distance
                mean_dist_vector = np.nanmean(dist_matrix_float, axis=1)
        elif self.cohort_ind_dict:
            """
            Compare different cohorts.
            """
            dist_matrix = cdist(self.cohort, self.ref_cohort, metric='braycurtis')
            dist_matrix_float = dist_matrix.astype(float)
            for key in self.cohort_ind_dict:
                for i in range(0, len(self.cohort_ind_dict[key])):
                    dist_matrix_float[key, self.cohort_ind_dict[key][i]] = np.nan
            if self.median:  # measure median distance
                mean_dist_vector = np.nanmedian(dist_matrix_float, axis=1)
            else:  # measure mean distance
                mean_dist_vector = np.nanmean(dist_matrix_float, axis=1)
        else:
            dist_matrix = cdist(self.cohort, self.ref_cohort, metric='braycurtis')
            if self.median:  # measure median distance
                mean_dist_vector = np.median(dist_matrix, axis=1)
            else:  # measure mean distance
                mean_dist_vector = np.mean(dist_matrix, axis=1)
        return mean_dist_vector

    def BC_distane(self):
        if self.cohort.ndim == 1:
            mean_dist_vector = self._BC_sample_vs_cohort()
        else:
            mean_dist_vector = self._BC_cohort_vs_cohort()
        return mean_dist_vector

    @staticmethod
    def _normalize_cohort(cohort):
        if cohort.ndim == 1:
            cohort_normalized = cohort / np.linalg.norm(cohort, ord=1)
            return cohort_normalized
        elif cohort.ndim == 2:
            cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
            return cohort_normalized
        else:
            raise ValueError("Invalid cohort dimension. Expected 1 or 2 dimensions.")