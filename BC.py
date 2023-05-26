import numpy as np
from scipy.spatial.distance import braycurtis
import math

def calc_bray_curtis_dissimilarity(first_cohort, second_cohort, second_cohort_ind_dict=np.nan,
                                   median=False, self_cohort=False):
    """
    :param first_cohort: The first cohort.
    :param second_cohort: The second cohort.
    :param median: If True, the function will calculate the median distance.
    :param self_cohort: If true, it means that first_cohort is identical to second_cohort and the function
     will not calculate distances to itself.
    :return: mean of median distance vector.
    """
    if second_cohort.ndim == 1:
        num_samples_first = np.size(first_cohort, 0)
        mean_dist_vector = 0
        if not math.isnan(second_cohort_ind_dict):
            k = second_cohort_ind_dict
            sample_dist = np.array([braycurtis(first_cohort[j, :], second_cohort) for j in range(
                0, num_samples_first) if j != k])
        else:
            sample_dist = np.array([braycurtis(first_cohort[j, :], second_cohort) for j in range(
                0, num_samples_first)])
        if median:  # measure median distance
            mean_dist_vector = np.median(sample_dist)
        else:  # measure mean distance
            mean_dist_vector = np.mean(sample_dist)
        return mean_dist_vector
    else:
        if self_cohort:
            """
            If we compare two identical cohorts, we want to avoid measuring distance between the same samples.
            """
            num_samples = np.size(first_cohort, 0)
            mean_dist_vector = np.zeros(num_samples)
            for i in range(0, num_samples):
                sample_dist = np.array([braycurtis(first_cohort[j, :], second_cohort[i, :]
                                                   ) for j in range(0, num_samples) if i != j])
                if median:  # measure median distance
                    mean_dist_vector[i] = np.median(sample_dist)
                else:  # measure mean distance
                    mean_dist_vector[i] = np.mean(sample_dist)
        else:
            """
            Compare different cohorts.
            """
            num_samples_first = np.size(first_cohort, 0)
            num_samples_second = np.size(second_cohort, 0)
            mean_dist_vector = np.zeros(num_samples_second)
            if isinstance(second_cohort_ind_dict, dict):
                for i in range(0, num_samples_second):
                    if i in second_cohort_ind_dict:
                        k = second_cohort_ind_dict[i]
                        sample_dist = np.array([braycurtis(first_cohort[j, :], second_cohort[i, :]
                                                           ) for j in range(0, num_samples_first) if j != k])
                    else:
                        sample_dist = np.array([braycurtis(first_cohort[j, :], second_cohort[i, :]
                                                           ) for j in range(0, num_samples_first)])
                    if median:  # measure median distance
                        mean_dist_vector[i] = np.median(sample_dist)
                    else:  # measure mean distance
                        mean_dist_vector[i] = np.mean(sample_dist)
            else:
                for i in range(0, num_samples_second):
                    sample_dist = np.array([braycurtis(first_cohort[j, :], second_cohort[i, :]
                                            ) for j in range(0, num_samples_first)])
                    if median:  # measure median distance
                        mean_dist_vector[i] = np.median(sample_dist)
                    else:  # measure mean distance
                        mean_dist_vector[i] = np.mean(sample_dist)
        return mean_dist_vector


