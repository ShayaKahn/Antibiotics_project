import numpy as np
from scipy.spatial.distance import braycurtis

class NonV:
    def __init__(self, Baseline_sample, ABX_set, Future_sample, Baseline_cohort, mean_num=100):
        self.Baseline_sample = Baseline_sample
        self.ABX_set = ABX_set
        self.Future_sample = Future_sample
        self.Baseline_cohort = Baseline_cohort
        self.intersection = self._find_intersection()
        self.vanishing_index = self._find_vanish()
        self.baseline_sub_sample, self.future_sub_sample = self._create_sub_samples()
        self.baseline_sub_sample = self._normalize_sample(self.baseline_sub_sample)
        self.future_sub_sample = self._normalize_sample(self.future_sub_sample)
        self.sub_baseline_cohort = self._create_sub_baseline_cohort()
        self.shuffled_sample_list = []
        for i in range(mean_num):
            self.shuffled_sample_list.append(self._normalize_sample(self._create_shuffled_sample()))

    def _find_intersection(self):
        non_zero_baseline_sample = np.nonzero(self.Baseline_sample)
        non_zero_future_sample = np.nonzero(self.Future_sample)
        intersection = np.intersect1d(non_zero_baseline_sample, non_zero_future_sample)
        return intersection

    def _find_vanish(self):
        if self.ABX_set.ndim != 1:
            vanishing_index = np.nonzero(np.any(self.ABX_set == 0, axis=0))[0]
        else:
            vanishing_index = np.where(self.ABX_set == 0)[0]
        return vanishing_index

    def _create_sub_samples(self):
        baseline_sub_sample = self.Baseline_sample[self.vanishing_index]
        future_sub_sample = self.Future_sample[self.vanishing_index]
        return baseline_sub_sample, future_sub_sample

    def _create_sub_baseline_cohort(self):
        sub_baseline_cohort = self.Baseline_cohort[:, self.vanishing_index]
        return sub_baseline_cohort

    def _create_shuffled_sample(self):
        n_rows, m_cols = self.sub_baseline_cohort.shape
        shuffled_sample = np.array([np.random.choice(self.sub_baseline_cohort[:, col]) for col in range(m_cols)])
        return shuffled_sample

    @staticmethod
    def _normalize_sample(sample):
        if np.all(sample == 0):
            return sample
        else:
            sample_normalized = sample / np.sum(sample)
        return sample_normalized

    def calculate_BC(self):
        bc_real = braycurtis(self.baseline_sub_sample, self.future_sub_sample)
        bc_shuffled_mean = np.mean([braycurtis(
            self.baseline_sub_sample, sample) for sample in self.shuffled_sample_list])
        return bc_real, bc_shuffled_mean
