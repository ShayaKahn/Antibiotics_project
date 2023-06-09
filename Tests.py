import unittest
from optimal_cohort import OptimalCohort
from idoa import IDOA
import numpy as np
from BrayCurtis import BC
from BC import calc_bray_curtis_dissimilarity
import pandas as pd
from OTU_fit import OtuFit
from IDOA_D_after_perturbation import IDOA_D
from GLV_model import GLV
from overlap import Overlap
import matplotlib.pyplot as plt

class TestOptimalCohort(unittest.TestCase):
    """
    This class tests the OptimalCohort class.
    """
    def setUp(self) -> None:
        # Two default samples.
        self.samples_dict = {'a': np.array([[11, 0, 8], [3, 9, 2], [0, 1, 3]]),
                             'b': np.array([[7, 1, 2], [1, 6, 0], [2, 3, 8], [8, 2, 5], [0, 1, 0]]),
                             'c': np.array([[35, 0, 17], [3, 4, 3], [1, 0, 8]]),
                             'd': np.array([[12, 7, 4], [1, 0, 0], [7, 1, 0], [6, 6, 6]])}
        self.optimal = OptimalCohort(self.samples_dict)
        self.optimal_samples = self.optimal.get_optimal_samples()
    def test_get_optimal_samples(self):
        self.assertEqual(np.sum(self.optimal_samples, axis=1).tolist(),
                         np.ones(np.size(self.optimal_samples, axis=0)).tolist())

class TestIDOA(unittest.TestCase):
    """
    This class tests the IDOA class.
    """
    def setUp(self) -> None:
        self.ref_cohort = np.array([[1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01],
                                    [11, 0, 1, 13, 0, 5, 0.1, 8.5, 4, 0],
                                    [2, 2, 1, 0, 10, 0, 0, 0, 4, 0.001],
                                    [30, 0, 0, 1, 0, 0, 0.18, 5, 4, 0],
                                    [30, 0, 0, 1, 0, 0, 0.18, 5, 4, 0],
                                    [13, 1, 1, 1, 3, 4, 0.1, 15, 1, 9],
                                    [8, 2, 4, 5, 6, 8, 0, 5, 3, 3],
                                    [3, 1, 3, 1, 4, 7, 2, 50, 3, 1]])

        self.single_sample_included = np.array([1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01])

        self.single_sample_not_included = np.array([4, 0, 0, 18, 1, 0, 0, 2, 0, 80])

        self.cohort_included = np.array([[1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01],
                                         [1, 10, 20, 0, 4, 8, 0.01, 19, 0, 0]])

        self.cohort_not_included = np.array([[2, 13, 1, 0, 0, 0, 0, 3, 0, 0.1],
                                             [1, 10, 20, 0, 4, 8, 0.01, 19, 0, 0]])

        self.idoa_ss_included = IDOA(self.ref_cohort, self.single_sample_included, min_overlap=0.2, max_overlap=1,
                                     zero_overlap=0, pos=True, identical=False)
        self.idoa_ss_included_vector = self.idoa_ss_included.calc_idoa_vector(second_cohort_ind_dict=(0,))

        self.idoa_ss_not_included = IDOA(self.ref_cohort, self.single_sample_not_included, min_overlap=0.2,
                                         max_overlap=1, zero_overlap=0, pos=True, identical=False)
        self.idoa_ss_not_included_vector = self.idoa_ss_not_included.calc_idoa_vector()

        self.idoa_cohort_included = IDOA(self.ref_cohort, self.cohort_included, min_overlap=0.2,
                                         max_overlap=1, zero_overlap=0, pos=True, identical=False)
        self.idoa_cohort_included_vector = self.idoa_cohort_included.calc_idoa_vector(second_cohort_ind_dict={0: (0,)})

        self.idoa_cohort_not_included = IDOA(self.ref_cohort, self.cohort_not_included, min_overlap=0.2,
                                             max_overlap=1, zero_overlap=0, pos=True, identical=False)
        self.idoa_cohort_not_included_vector = self.idoa_cohort_not_included.calc_idoa_vector()

        self.idoa_identical = IDOA(self.ref_cohort, self.ref_cohort, min_overlap=0.2,
                                   max_overlap=1, zero_overlap=0, pos=True, identical=True)
        self.idoa_identical_vector = self.idoa_identical.calc_idoa_vector()

    def test_calc_idoa_vector(self):

        # Test single sample included
        self.assertEqual(np.size(self.idoa_ss_included_vector), 1)
        self.assertEqual(np.size(self.idoa_ss_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0) - 1)
        self.assertFalse(np.any(self.idoa_ss_included.dissimilarity_overlap_container_no_constraint[0][0] == 1),
                         "Overlap contains 1")
        self.assertFalse(np.any(self.idoa_ss_included.dissimilarity_overlap_container_no_constraint[0][1] == 0),
                         "Dissimilarity contains 0")

        # Test single sample not included
        self.assertEqual(np.size(self.idoa_ss_not_included_vector), 1)
        self.assertEqual(np.size(self.idoa_ss_not_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0))

        # Test cohort included
        self.assertEqual(np.size(self.idoa_cohort_included_vector), np.size(self.cohort_included, axis=0))
        self.assertEqual(np.size(self.idoa_cohort_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0) - 1)
        self.assertFalse(np.any(self.idoa_cohort_included.dissimilarity_overlap_container_no_constraint[0][0] == 1),
                         "Overlap contains 1")
        self.assertFalse(np.any(self.idoa_cohort_included.dissimilarity_overlap_container_no_constraint[0][1] == 0),
                         "Dissimilarity contains 0")

        # Test cohort not included
        self.assertEqual(np.size(self.idoa_cohort_not_included_vector),
                         np.size(self.idoa_cohort_not_included_vector, axis=0))
        self.assertEqual(np.size(self.idoa_cohort_not_included.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0))

        # Test identical
        self.assertEqual(np.size(self.idoa_identical_vector), np.size(self.ref_cohort, axis=0))
        self.assertEqual(np.size(self.idoa_identical.dissimilarity_overlap_container_no_constraint[0][0]),
                         np.size(self.ref_cohort, axis=0) - 1)
        self.assertFalse(np.any(self.idoa_identical.dissimilarity_overlap_container_no_constraint[0][0] == 1),
                         "Overlap contains 1")
        self.assertFalse(np.any(self.idoa_identical.dissimilarity_overlap_container_no_constraint[0][1] == 0),
                         "Dissimilarity contains 0")

class TestBC(unittest.TestCase):
    """
    This class tests the BC class.
    """
    def setUp(self) -> None:

        def _normalize_cohort(cohort):
            if cohort.ndim == 1:
                cohort_normalized = cohort / np.linalg.norm(cohort, ord=1)
                return cohort_normalized
            elif cohort.ndim == 2:
                cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
                return cohort_normalized
            else:
                raise ValueError("Invalid cohort dimension. Expected 1 or 2 dimensions.")

        self.ref_cohort = np.array([[1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01],
                                    [11, 0, 1, 13, 0, 5, 0.1, 8.5, 4, 0],
                                    [2, 2, 1, 0, 10, 0, 0, 0, 4, 0.001],
                                    [30, 0, 0, 1, 0, 0, 0.18, 5, 4, 0],
                                    [30, 0, 0, 1, 0, 0, 0.18, 5, 4, 0],
                                    [13, 1, 1, 1, 3, 4, 0.1, 15, 1, 9],
                                    [8, 2, 4, 5, 6, 8, 0, 5, 3, 3],
                                    [3, 1, 3, 1, 4, 7, 2, 50, 3, 1]])

        self.single_sample_included = np.array([1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01])

        self.single_sample_not_included = np.array([4, 0, 0, 18, 1, 0, 0, 2, 0, 80])

        self.cohort_included = np.array([[1, 3, 0, 7, 14, 0, 0.5, 8, 44, 0.01],
                                         [1, 10, 20, 0, 4, 8, 0.01, 19, 0, 0]])

        self.cohort_not_included = np.array([[2, 13, 1, 0, 0, 0, 0, 3, 0, 0.1],
                                             [1, 10, 20, 0, 4, 8, 0.01, 19, 0, 0]])

        self.check_cohort = np.array([[0.2, 0.6, 0.2],
                                      [0.3, 0.3, 0.4],
                                      [0, 0, 1]])

        self.check_ref_cohort = np.array([[0.1, 0.1, 0.8],
                                          [0.5, 0, 0.5],
                                          [0.6, 0.4, 0]])

        self.check = BC(self.check_ref_cohort, self.check_cohort, identical=True, median=False)
        self.check_vector = self.check.BC_distane()
        print(self.check_vector)

        self.bc_ss_included = BC(self.ref_cohort, self.single_sample_included,
                                 cohort_ind_dict=(0, ))
        self.bc_ss_included_vector = self.bc_ss_included.BC_distane()
        self.bc_ss_included_vector_test = calc_bray_curtis_dissimilarity(
            _normalize_cohort(self.ref_cohort), _normalize_cohort(self.single_sample_included),
            second_cohort_ind_dict=0)

        self.bc_ss_not_included = BC(self.ref_cohort, self.single_sample_not_included)
        self.bc_ss_not_included_vector = self.bc_ss_not_included.BC_distane()
        self.bc_ss_not_included_vector_test = calc_bray_curtis_dissimilarity(
            _normalize_cohort(self.ref_cohort), _normalize_cohort(self.single_sample_not_included))

        self.bc_cohort_included = BC(self.ref_cohort, self.cohort_included,
                                     cohort_ind_dict={0: (0,)})
        self.bc_cohort_included_vector = self.bc_cohort_included.BC_distane()
        self.bc_cohort_included_vector_test = calc_bray_curtis_dissimilarity(
            _normalize_cohort(self.ref_cohort), _normalize_cohort(self.cohort_included),
            second_cohort_ind_dict={0: 0})

        self.bc_cohort_not_included = BC(self.ref_cohort, self.cohort_not_included)
        self.bc_cohort_not_included_vector = self.bc_cohort_not_included.BC_distane()
        self.bc_cohort_not_included_vector_test = calc_bray_curtis_dissimilarity(
            _normalize_cohort(self.ref_cohort), _normalize_cohort(self.cohort_not_included))

        self.bc_identical = BC(self.ref_cohort, self.ref_cohort, identical=True)
        self.bc_identical_vector = self.bc_identical.BC_distane()
        self.bc_identical_vector_test = calc_bray_curtis_dissimilarity(
            first_cohort=_normalize_cohort(self.ref_cohort),
            second_cohort=_normalize_cohort(self.ref_cohort), self_cohort=True)

    def test_BC_distance(self):
        places = 5
        self.assertEqual(float(np.mean(self.bc_ss_included_vector -
                                       self.bc_ss_included_vector_test)), 0, places)
        self.assertAlmostEqual(float(np.mean(
            self.bc_ss_not_included_vector - self.bc_ss_not_included_vector_test)), 0, places)
        self.assertAlmostEqual(float(np.mean(
            self.bc_cohort_included_vector - self.bc_cohort_included_vector_test)), 0, places)
        self.assertAlmostEqual(float(np.mean(self.bc_cohort_not_included_vector -
                                             self.bc_cohort_not_included_vector_test)), 0, places)
        self.assertAlmostEqual(float(np.mean(self.bc_identical_vector -
                               self.bc_identical_vector_test)), 0, places)

class TestOTUFit(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_cohort = pd.DataFrame({'Species': ['A', 'B', 'D', 'C'], 1: [0.3, 0.3, 0.1, 0.3],
                                        2: [0.5, 0.5, 0, 0], 3: [0.1, 0, 0.8, 0.1],
                                        4: [1, 0, 0, 0]})

        self.cohort = pd.DataFrame({'Species': ['B', 'C', 'D'], 1: [0.4, 0.3, 0.3],
                                    2: [1, 0, 0], 3: [0, 0.2, 0.8], 4: [0, 0.5, 0.5]})

        print(self.ref_cohort)
        print(self.cohort)

        self.OtuPipeline_object = OtuFit(self.ref_cohort, self.cohort, threshold=0.99)

    def test_pipe(self):
        sub_ref_OTU, sub_OTU = self.OtuPipeline_object.pipe()
        print(self.OtuPipeline_object.fraction_ref_OTU)
        print(self.OtuPipeline_object.fraction_OTU)

class Test_IDOA_D(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = np.array([0.1, 0, 0.2, 0.4, 0, 0, 0.1, 0.2])
        self.cohort = np.array([[0, 0.1, 0.3, 0.4, 0.1, 0.1, 0, 0],
                                [0.1, 0.1, 0.1, 0.3, 0.2, 0, 0, 0.2],
                                [0.5, 0, 0.2, 0.2, 0, 0, 0.1, 0]])
        print(np.shape(self.cohort))
        print(np.shape(self.sample))

    def test_calculate_values(self):
        IDOA_D_object = IDOA_D(self.sample, self.cohort, min_overlap=0.5, max_overlap=1, zero_overlap=0.1, pos=False,
                identical=False, min_num_points=0, percentage=None, ind=(0, ), median=False)
        print(IDOA_D_object.calculate_values())

class Test_GLV(unittest.TestCase):
    def setUp(self) -> None:
        self.glv = GLV(n_samples=100, n_species=100, delta=1e-2, final_time=1,
                       max_step=0.2, p_mat=0.1, p_init=0.8, p_alt_init=0.2, sigma=0.2)

    def test_solve(self):
        final_abundances = self.glv.solve()
        final_abundances_perturb = self.glv.solve(perturbation=True)
        print(final_abundances)
        print(final_abundances_perturb)

class Test_Overlap(unittest.TestCase):
    def setUp(self) -> None:
        self.first_sample = np.array([0.1, 0, 0.2, 0.4, 0, 0, 0.1, 0.2])
        self.second_sample = np.array([0.1, 0, 0.2, 0.4, 0.1, 0, 0, 0.2])

    def test_calculate_overlap(self):
        overlap = Overlap(self.first_sample, self.second_sample, overlap_type='Jaccard').calculate_overlap()
        print(overlap)

