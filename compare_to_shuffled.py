import numpy as np
from scipy.spatial.distance import braycurtis
from overlap import Overlap

class ShuffledVsNormal:
    def __init__(self, Baseline_sample, ABX_sample, Future_sample, Baseline_cohort, index, strict=False, mean_num=100):
        self.Baseline_sample = Baseline_sample
        self.ABX_sample = ABX_sample
        self.Future_sample = Future_sample
        self.strict = strict
        self.Baseline_cohort = np.delete(Baseline_cohort, index, axis=0)
        self.intersection = self._find_intersection()
        self.vanishing_index = self._find_vanish()
        self.baseline_sub_sample, self.future_sub_sample = self._create_sub_samples()
        self.baseline_sub_sample = self._normalize_sample(self.baseline_sub_sample)
        self.future_sub_sample = self._normalize_sample(self.future_sub_sample)
        self.sub_baseline_cohort = self._create_sub_baseline_cohort()
        self.shuffled_sample_list = [self._normalize_sample(self._create_shuffled_sample()) for _ in range(mean_num)]

    def _find_intersection(self):
        non_zero_baseline_sample = np.nonzero(self.Baseline_sample)
        non_zero_future_sample = np.nonzero(self.Future_sample)
        intersection = np.intersect1d(non_zero_baseline_sample, non_zero_future_sample)
        return intersection

    def _find_vanish(self):
        if self.strict:
            nonzero_ABX = np.nonzero(self.ABX_sample)
            nonzero_base = np.nonzero(self.Baseline_sample)
            nonzero_future = np.nonzero(self.Future_sample)
            intersect_ABX_base = np.intersect1d(nonzero_ABX, nonzero_base)
            intersect_ABX_future = np.intersect1d(nonzero_ABX, nonzero_future)
            intersect_of_intersect = np.intersect1d(intersect_ABX_base, intersect_ABX_future)
            vanishing_index = np.setdiff1d(np.arange(len(self.Baseline_sample)), intersect_of_intersect)
        else:
            vanishing_index = np.where(self.ABX_sample == 0)[0]
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

    def BC(self):
        bc_real = braycurtis(self.baseline_sub_sample, self.future_sub_sample)
        bc_shuffled_mean = np.mean([braycurtis(
            self.future_sub_sample, sample) for sample in self.shuffled_sample_list])
        return bc_real, bc_shuffled_mean

    def Jaccard(self):
        jaccard_real = Overlap(self.baseline_sub_sample,
                               self.future_sub_sample, overlap_type="Jaccard").calculate_overlap()
        jaccard_shuffled_mean = np.mean([Overlap(self.future_sub_sample, sample, overlap_type="Jaccard"
                                                 ).calculate_overlap() for sample in self.shuffled_sample_list])
        return jaccard_real, jaccard_shuffled_mean

