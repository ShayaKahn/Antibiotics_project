from overlap import Overlap
import numpy as np

class JaccardDisappearedSpecies:
    def __init__(self, baseline_sample, future_sample, ABX_sample, strict=False):
        self.base_sample = baseline_sample
        self.future_sample = future_sample
        self.ABX_sample = ABX_sample
        if strict:
            self.nonzero_ABX = np.nonzero(self.ABX_sample)
            self.nonzero_base = np.nonzero(self.base_sample)
            self.nonzero_future = np.nonzero(self.future_sample)
            self.intersect_ABX_base = np.intersect1d(self.nonzero_ABX, self.nonzero_base)
            self.intersect_ABX_future = np.intersect1d(self.nonzero_ABX, self.nonzero_future)
            self.intersect_of_intersect = np.intersect1d(self.intersect_ABX_base, self.intersect_ABX_future)
            self.non_ARS = np.setdiff1d(np.arange(len(self.base_sample)), self.intersect_of_intersect)
        else:
            self.non_ARS = np.where(self.ABX_sample == 0)
        self.sub_base_sample = self.base_sample[self.non_ARS]
        self.sub_future_sample = self.future_sample[self.non_ARS]
        self.jaccard_object_subject = Overlap(self.sub_base_sample, self.sub_future_sample, overlap_type="Jaccard")
        self.jaccard_subject = self.jaccard_object_subject.calculate_overlap()

    def calc_jaccard(self, external_sample_base):
        sub_external_sample_base = external_sample_base[self.non_ARS]
        jaccard_object_external = Overlap(self.sub_future_sample, sub_external_sample_base, overlap_type="Jaccard")
        jaccard_external = jaccard_object_external.calculate_overlap()
        return jaccard_external


