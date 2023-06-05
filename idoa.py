import numpy as np
from overlap import Overlap
from dissimilarity import Dissimilarity
from scipy.stats import linregress

class IDOA:
    """
    This class calculates the IDOA values vector for a cohort or the IDOA value for a sample
    with respect to a reference cohort.
    """
    def __init__(self, ref_cohort, cohort, min_overlap=0.5, max_overlap=1, zero_overlap=0.1, pos=False,
                 identical=False, min_num_points=0, percentage=None):
        """
        :param ref_cohort: The reference cohort, samples are in the rows.
        :param cohort: The cohort, samples are in the rows.
        :param min_overlap: The minimal value of overlap.
        :param max_overlap: The maximal value of overlap.
        :param zero_overlap: A number, if the maximal value of the overlap vector that calculated
               between sample from the second cohort w.r.t the first cohort is less than min_overlap + zero_overlap
               so the overlap considered to be zero.
        :param pos: If true, positive slope considered as zero.
        :param identical: If True, both cohorts are considered as identical.
        :param min_num_points: The minimal number of points to calculate the IDOA.
        """
        self.ref_cohort = ref_cohort
        self.cohort = cohort
        self.min_num_points = min_num_points
        if type(self.min_num_points) is not int:
            raise ValueError("min_num_points should be a positive integer")
        if self.ref_cohort.ndim != 2:
            raise ValueError("ref_cohort should be a 2D numpy array")
        if self.cohort.ndim not in [1, 2]:
            raise ValueError("cohort should be a 1D or 2D numpy array")
        if self.cohort.ndim == 1 and self.cohort.size != self.ref_cohort.shape[1]:
            raise ValueError("The size of cohort should match the number of columns in ref_cohort")
        if self.cohort.ndim == 2 and self.cohort.shape[1] != self.ref_cohort.shape[1]:
            raise ValueError("The number of columns in cohort should match the number of columns in ref_cohort")
        try:
            self.min_overlap = min_overlap
            assert isinstance(self.min_overlap, (int, float))
            self.max_overlap = max_overlap
            assert isinstance(self.max_overlap, (int, float))
            self.zero_overlap = zero_overlap
            assert isinstance(self.zero_overlap, (int, float))
        except AssertionError:
            raise ValueError("Invalid input values for min_overlap, max_overlap or zero_overlap. "
                             "They should be numeric values")
        try:
            assert (0 <= min_overlap < 1)
            assert (min_overlap < max_overlap and (0 < max_overlap <= 1))
            assert (min_overlap + zero_overlap < max_overlap)
        except AssertionError:
            raise ValueError("Invalid input values for min_overlap, max_overlap or zero_overlap. "
                             "Their values should be between 0 and 1, with min_overlap less than max_overlap "
                             "and min_overlap plus zero_overlap less than max_overlap.")
        self.identical = identical
        self.pos = pos
        try:
            assert isinstance(self.identical, bool)
            assert isinstance(self.pos, bool)
        except AssertionError:
            raise  AssertionError("Invalid input values for identical or pos. They should be boolean values.")
        if self.identical and self.ref_cohort.shape != self.cohort.shape:
            raise ValueError("If identical=True, the dimensions of self.cohort and self.ref_cohort should be the same.")
        self.percentage = percentage
        if self.percentage:
            if not (isinstance(self.percentage, int) or isinstance(self.percentage, float)):
                raise ValueError('percentage should be int or float')
            if not (0 <= self.percentage <= 100):
                raise ValueError('percentage must take values between 0 and 100 inclusive.')
        self.num_samples_ref = ref_cohort.shape[0]
        self.num_samples_cohort = cohort.shape[0]
        self.IDOA_vector = 0 if self.cohort.ndim == 1 else np.zeros(self.num_samples_cohort)
        self.dissimilarity_overlap_container = []
        self.dissimilarity_overlap_container_no_constraint = []

    def _create_od_vectors(self, sample, index=(None,)):
        """
        :param sample: A sample
        :param index: Integer
        :return: overlap_vector and dissimilarity_vector, this vectors contain the overlap and dissimilarity values.
        """
        o_vector = []
        d_vector = []
        for j in range(0, self.num_samples_ref):
            o = Overlap(self.ref_cohort[j, :], sample)  # Initiate Overlap
            d = Dissimilarity(self.ref_cohort[j, :], sample)  # Initiate Dissimilarity
            o_vector.append(o)
            d_vector.append(d)
        overlap_vector = np.array([o_vector[j].calculate_overlap()
                                   for j in range(0, self.num_samples_ref) if j not in index])  # Calculate overlap vector
        dissimilarity_vector = np.array([d_vector[j].calculate_dissimilarity()
                                        for j in range(0, self.num_samples_ref) if j not in index])  # Calculate
        # dissimilarity vector
        return overlap_vector, dissimilarity_vector

    def _filter_od_vectors(self, overlap_vector, dissimilarity_vector):
        """
        :param overlap_vector: Vector that contains the overlap values
        :param dissimilarity_vector: Vector that contains the dissimilarity values
        :return: filtered_overlap_vector and filtered_dissimilarity_vector, the original vectors after filtering.
        """
        #####################################
        if self.percentage:
            overlap_vector_index = np.where(overlap_vector > np.percentile(overlap_vector, self.percentage))
        else:
            overlap_vector_index = np.where(np.logical_and(overlap_vector >= self.min_overlap,
                                                           overlap_vector <= self.max_overlap))
            if overlap_vector_index[0].size == 0:
                raise ValueError("No overlap values found within the given range")
        #####################################
        filtered_overlap_vector = overlap_vector[overlap_vector_index]
        filtered_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
        return filtered_overlap_vector, filtered_dissimilarity_vector

    def _calc_idoa_vector_sample_vs_cohort(self, ind):
        """
        This is a private method that calculates the IDOA value for single sample w.r.t a cohort
        :param ind: index
        :return: IDOA vector
        """
        if ind[0] is not None:
            k = ind
            overlap_vector, dissimilarity_vector = self._create_od_vectors(self.cohort, index=k)
        else:
            overlap_vector, dissimilarity_vector = self._create_od_vectors(self.cohort)
        self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector, dissimilarity_vector)))
        # Set IDOA as 0 for low overlap values
        if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
            self.IDOA_vector = 0
        else:
            filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                overlap_vector, dissimilarity_vector)
            self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                   filtered_dissimilarity_vector)))
            slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
            self.IDOA_vector = slope if not np.isnan(slope) else 0  # If the slope is a valid
            # number set: IDOA = slope
            if self.pos:  # If pos == True, we set positive IDOA to 0
                if self.IDOA_vector > 0:
                    self.IDOA_vector = 0
            if np.size(filtered_overlap_vector) < self.min_num_points:
                self.IDOA_vector = 0
            return self.IDOA_vector

    def _calc_idoa_vector_cohort_vs_cohort_identical(self):
        """
        This is a private method that calculates the IDOA vector for a cohort w.r.t to itself
        :return: IDOA vector
        """
        for i in range(0, self.num_samples_cohort):
            overlap_vector, dissimilarity_vector = self._create_od_vectors(
                self.cohort[i, :], index=[i])
            self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector,
                                                                                dissimilarity_vector)))
             # Set IDOA as 0 for low overlap values
            if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
                self.IDOA_vector[i] = 0
            else:
                filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                    overlap_vector, dissimilarity_vector)
                self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                       filtered_dissimilarity_vector)))
                slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
                self.IDOA_vector[i] = slope if not np.isnan(slope) else 0  # If the slope is a valid number set:
                # IDOA = slope
                if np.size(filtered_overlap_vector) < self.min_num_points:
                    self.IDOA_vector[i] = 0
        if self.pos:  # If pos == True, we set positive IDOA to 0
            self.IDOA_vector[self.IDOA_vector > 0] = 0
        return self.IDOA_vector

    def _calc_idoa_vector_cohort_vs_cohort_custom(self, ind):
        """
        This is a private method that calculates the IDOA vector for a cohort w.r.t to a different cohort and includes
        constraints on specific indexes
        :param ind: dictionary of indexes, if i --> j it means that the dissimilarity and overlap are not calculated for
         sample i w.r.t sample j
        :return: IDOA vector
        """
        for i in range(0, self.num_samples_cohort):
            if i in ind:
                k = ind[i]
                overlap_vector, dissimilarity_vector = self._create_od_vectors(
                    self.cohort[i, :], index=k)
            else:
                overlap_vector, dissimilarity_vector = self._create_od_vectors(
                    self.cohort[i, :])
            self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector,
                                                                                dissimilarity_vector)))
            # Set IDOA as 0 for low overlap values
            if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
                self.IDOA_vector[i] = 0
            else:
                filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                    overlap_vector, dissimilarity_vector)
                self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                       filtered_dissimilarity_vector)))
                slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
                self.IDOA_vector[i] = slope if not np.isnan(slope) else 0  # If the slope is a valid number set:
                # IDOA = slope
                if np.size(filtered_overlap_vector) < self.min_num_points:
                    self.IDOA_vector[i] = 0
        if self.pos:  # If pos == True, we set positive IDOA to 0
            self.IDOA_vector[self.IDOA_vector > 0] = 0
        return self.IDOA_vector

    def _calc_idoa_vector_cohort_vs_cohort_not_identical(self):
        """
        This is a private method that calculates the IDOA vector for a cohort w.r.t to a different cohort
        :return: IDOA
        """
        for i in range(0, self.num_samples_cohort):
            overlap_vector, dissimilarity_vector = self._create_od_vectors(
                self.cohort[i, :])
            self.dissimilarity_overlap_container_no_constraint.append(np.vstack((overlap_vector,
                                                                                 dissimilarity_vector)))
            # Set IDOA as 0 for low overlap values
            if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap and not self.percentage:
                self.IDOA_vector[i] = 0
            else:
                filtered_overlap_vector, filtered_dissimilarity_vector = self._filter_od_vectors(
                    overlap_vector, dissimilarity_vector)
                self.dissimilarity_overlap_container.append(np.vstack((filtered_overlap_vector,
                                                                       filtered_dissimilarity_vector)))
                slope = linregress(filtered_overlap_vector, filtered_dissimilarity_vector)[0]  # Calculate the slope
                self.IDOA_vector[i] = slope if not np.isnan(slope) else 0  # If the slope is a valid number set:
                # IDOA = slope
                if np.size(filtered_overlap_vector) < self.min_num_points:
                    self.IDOA_vector[i] = 0
        if self.pos:  # If pos == True, we set positive IDOA to 0
            self.IDOA_vector[self.IDOA_vector > 0] = 0
        return self.IDOA_vector

    def calc_idoa_vector(self, second_cohort_ind_dict=(None,)):
        """
        This method calculates the vector of the IDOA values that calculated for a cohort of samples w.r.t the
         reference cohort for all the optional cases(identical or not, single sample or not).
        :return: IDOA vector.
        """
        if self.cohort.ndim == 1:  # Check if the cohort is a single sample
            """
            if not isinstance(second_cohort_ind_dict, dict):
                raise ValueError('second_cohort_ind_dict must be a dictionary')
            else:
                for key, value in second_cohort_ind_dict.items():
                    if not isinstance(key, int) or not isinstance(value, tuple):
                        raise ValueError('The keys of the dictionary must be integers and the values must be'
                                         'tuples')
                    for item in value:
                        if not isinstance(item, int):
                            raise ValueError('the values must be tuples that contain integers')
            """
            return self._calc_idoa_vector_sample_vs_cohort(ind=second_cohort_ind_dict)
        else:
            if self.identical:  # Check if the cohorts considered to be identical
                return self._calc_idoa_vector_cohort_vs_cohort_identical()
            else:
                if second_cohort_ind_dict[0] is not None:  # Check if index dictionary is available
                    return self._calc_idoa_vector_cohort_vs_cohort_custom(ind=second_cohort_ind_dict)
                else:
                    return self._calc_idoa_vector_cohort_vs_cohort_not_identical()