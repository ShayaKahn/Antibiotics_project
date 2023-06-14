import numpy as np
from dissimilarity import Dissimilarity
from idoa import IDOA
from BrayCurtis import BC

class IDOA_D:
    def __init__(self, sample, baseline_cohort, min_overlap=0.5, max_overlap=1, zero_overlap=0.1, pos=False,
                 identical=False, min_num_points=0, percentage=None, ind=None, median=False):
        self.sample = sample
        self.baseline_cohort = baseline_cohort
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.zero_overlap = zero_overlap
        self.pos = pos
        self.identical = identical
        self.min_num_points = min_num_points
        self.percentage = percentage
        self.ind = ind
        self.median = median
        self.new_cohort, self.new_sample = self._build_new_cohort_sample()
        self.d_o_container = None

    def _find_non_zero(self):
        non_zero_index = np.nonzero(self.sample)
        return non_zero_index

    def _build_new_cohort_sample(self):
        non_zero_index = self._find_non_zero()
        new_sample = self.sample[non_zero_index[0]]
        new_cohort = self.baseline_cohort[:, non_zero_index[0]]
        return new_cohort, new_sample

    def _calculate_dissimilarity(self):
        mean_D = np.mean([Dissimilarity(self.new_sample, self.new_cohort[i, :]
                           ).calculate_dissimilarity() for i in range(self.new_cohort.shape[0])])
        return mean_D

    def _calculate_idoa(self):
        IDOA_object = IDOA(self.new_cohort, self.new_sample, self.min_overlap, self.max_overlap,
                        self.zero_overlap, self.pos, self.identical, self.min_num_points,
                        self.percentage)
        IDOA_val = IDOA_object.calc_idoa_vector(self.ind)
        self.d_o_container = IDOA_object.dissimilarity_overlap_container_no_constraint
        return IDOA_val

    def _calculate_BC(self):
        mean_BC_val = BC(self.new_cohort, self.new_sample, self.ind, self.median, self.identical).BC_distane()
        return mean_BC_val

    def calculate_values(self):
        D_value = self._calculate_dissimilarity()
        IDOA_value = self._calculate_idoa()
        BC_value = self._calculate_BC()
        return IDOA_value, D_value, BC_value
