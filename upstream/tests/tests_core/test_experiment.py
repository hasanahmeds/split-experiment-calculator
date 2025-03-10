from __future__ import division # so that, under Python 2.*, integer division results in real numbers

import unittest
import warnings

import numpy as np

from expan.core.results import CombinedTestStatistics
from expan.core.statistical_test import *
from expan.core.experiment import Experiment
from expan.core.util import generate_random_data, find_value_by_key_with_condition


class ExperimentTestCase(unittest.TestCase):
    """ Defines the setUp() and tearDown() functions for the statistics test cases."""

    def setUp(self):
        """ Load the needed data sets for all StatisticsTestCases and set the random
        seed so that randomized algorithms show deterministic behaviour."""
        np.random.seed(0)
        data, metadata = generate_random_data()

        self.data, self.metadata = data, metadata

        feature_has = FeatureFilter('feature', 'has')
        feature_non = FeatureFilter('feature', 'non')
        feature_one = FeatureFilter('feature', 'feature that only has one data point')

        # simple statistical test
        self.kpi = KPI('normal_same')
        self.variants = Variants('variant', 'B', 'A')
        self.nonsense_variants = Variants('variant', 'C', 'D')
        self.test_normal_same = StatisticalTest(self.data, self.kpi, [], self.variants)
        self.test_nonsense_variant = StatisticalTest(self.data, self.kpi, [], self.nonsense_variants)
        self.test_normal_same_feature_non = StatisticalTest(self.data, self.kpi, [feature_non], self.variants)
        self.test_normal_same_feature_has = StatisticalTest(self.data, self.kpi, [feature_has], self.variants)
        self.test_normal_same_feature_one = StatisticalTest(self.data, self.kpi, [feature_one], self.variants)

        # statistical test with derived kpi
        self.derived_kpi = DerivedKPI('derived_kpi_one', 'normal_same', 'normal_shifted')
        self.test_derived_kpi = StatisticalTest(self.data, self.derived_kpi, [], self.variants)

        # small dummy data frame
        data_dummy = np.array([['index', 'entity', 'variant', 'normal_same', 'normal_shifted'],
                               [0, 1, 'A', 2.0, 1.0], [1, 2, 'B', 3.0, 2.0],
                               [2, 3, 'A', 3.0, 2.0], [3, 4, 'B', 2.5, 1.0]])
        self.data_dummy_df = pd.DataFrame(data=data_dummy[1:, 1:],
                                          columns=data_dummy[0, 1:]).convert_objects(convert_numeric=True)

        data_dummy_with_nan = np.array([['index', 'entity', 'variant', 'normal_same', 'normal_shifted'],
                               [0, 1, 'A', 2.0, 1.0], [1, 2, 'B', 3.0, 2.0],
                               [2, 3, 'A', 3.0, 2.0], [3, 4, 'B', 2.5, 1.0],
                               [4, 5, 'A', 0.0, 0.0], [5, 6, 'B', None, None]])
        self.data_dummy_df_with_nan = pd.DataFrame(data=data_dummy_with_nan[1:, 1:],
                                                   columns=data_dummy_with_nan[0, 1:]).convert_objects(convert_numeric=True)

        # statistical test suite
        self.suite_with_one_test = StatisticalTestSuite([self.test_normal_same])
        self.suite_with_one_test_correction = StatisticalTestSuite([self.test_normal_same], CorrectionMethod.BH)
        self.suite_with_two_tests = StatisticalTestSuite([self.test_normal_same, self.test_derived_kpi],
                                                         CorrectionMethod.BONFERRONI)
        self.suite_with_one_subgroup = StatisticalTestSuite([self.test_normal_same_feature_one])
        self.suite_with_three_subgroups = StatisticalTestSuite([self.test_normal_same_feature_non,
                                                                self.test_normal_same_feature_has,
                                                                self.test_normal_same_feature_one],
                                                               CorrectionMethod.BH)

        # small dummy data frames producing zero std and -1 statistical power
        data_dummy_zero_std = np.array([['index', 'entity', 'variant', 'normal_same', 'normal_shifted'],
                                        [0, 1, 'A', 1.0, 1.0], [1, 2, 'B', 2.0, 2.0],
                                        [2, 3, 'A', 1.0, 1.0], [3, 4, 'B', 2.0, 2.0]])
        self.data_dummy_zero_std = pd.DataFrame(data=data_dummy_zero_std[1:, 1:],
                                                columns=data_dummy_zero_std[0, 1:]).convert_objects(
            convert_numeric=True)
        self.test_normal_same_zero_std = StatisticalTest(self.data_dummy_zero_std, self.kpi, [], self.variants)
        self.suite_with_one_test_zero_std = StatisticalTestSuite([self.test_normal_same_zero_std])

        # small dummy data frames with all nan values
        self.data_dummy_all_nan = pd.DataFrame({
                        'entity' : [1,2,3,4],
                        'variant' : ['A','B','C','D'],
                        'normal_same' : [np.nan] * 4,
                        'normal_shifted' : [np.nan] * 4,
                            })
        self.test_normal_same_nan_data = StatisticalTest(self.data_dummy_all_nan, self.kpi, [], self.variants)
        self.suite_with_one_test_with_nan_data = StatisticalTestSuite([self.test_normal_same_nan_data])

    def tearDown(self):
        """ Clean up after the test """
        pass

    def getExperiment(self):
        return Experiment(self.metadata)


class ExperimentClassTestCases(ExperimentTestCase):

    def test_Experiment_constructor(self):
        experiment = self.getExperiment()
        self.assertTrue(isinstance(experiment.metadata, dict))


class StatisticalTestTestCases(ExperimentTestCase):
    """ Test the method analyze_statistical_test. """

    def test_fixed_horizon(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same, test_method='fixed_horizon')

        self.assertAlmostEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertAlmostEqual(lower_bound_ci, -0.007135, ndecimals)
        self.assertAlmostEqual(upper_bound_ci, 0.073240, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size,   3892)

        self.assertAlmostEqual(res.result.treatment_statistics.mean,  0.025219, ndecimals)
