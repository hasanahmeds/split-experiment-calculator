import unittest
import warnings

import numpy as np
from scipy import stats

import expan.core.statistics as statx
from expan.core.util import find_value_by_key_with_condition
from .util import *

warnings.simplefilter('always')

class StatisticsTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data'
        self.samples = get_norm_temp_data(data_dir)
        self.metrics = get_framingham_data(data_dir)
        self.rand_s1 = np.random.normal(loc=0, size=1000)
        self.rand_s2 = np.random.normal(loc=0.1, size=1000)

    def tearDown(self):
        pass


class DeltaTestCases(StatisticsTestCase):
    def test__delta__not_providing_data_fails(self):
        """ Value error raised when not providing data. """
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.delta(self.samples.temperature, None)
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.delta(None, self.samples.temperature)

    def test__delta__computation_assumed_normal(self):
        """ Result of delta() assuming normality equals expected result. """
        # Computing delta assumed normal
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            assume_normal=True)
        # Checking if mean has right value
        self.assertAlmostEqual(res.delta, -0.28923076923075541)

        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')

        # Checking if lower percentile has right value
        self.assertAlmostEqual(value025, -0.53963938185557114)
        # Checking if uper percentile has right value
        self.assertAlmostEqual(value975, -0.038822156605939739)
        # Checking if sample size 1 is correct
        self.assertEqual(res.treatment_statistics.sample_size, 65)
        # Checking if sample size 2 is correct
        self.assertEqual(res.control_statistics.sample_size, 65)

    def test__delta__2percentiles_no_tests(self):
        """ Percentiles of delta() for sga are corrected for no tests (1 as a default). """
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2])

        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')

        self.assertAlmostEqual(value025, -0.53963938185557114)
        self.assertAlmostEqual(value975, -0.038822156605939739)

    def test__delta__nan_handling(self):
        """ Test correct handling of nans. (ignored). """
        res = statx.delta(self.rand_s1, self.rand_s2)
        self.assertEqual(res.treatment_statistics.sample_size, 1000)
        self.assertEqual(res.control_statistics.sample_size, 1000)

        r1 = self.rand_s1.copy()
        r1[90:] = np.nan
        res = statx.delta(r1, self.rand_s2)

        self.assertAlmostEqual (res.delta, -0.1, 1)
        self.assertEqual       (res.treatment_statistics.sample_size, 90)
        self.assertEqual       (res.control_statistics.sample_size, 1000)

    def test__delta__computation_not_assumed_normal(self):
        """ Result of delta() not assuming normality equals expected result. """
        # Computing delta not assumed normal
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            assume_normal=True)
        # Checking if mean has right value
        self.assertAlmostEqual(res.delta, -0.28923076923075541)

        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')

        # Checking if lower percentile has right value
        self.assertAlmostEqual(value025, -0.53963938185557114)
        # Checking if uper percentile has right value
        self.assertAlmostEqual(value975, -0.038822156605939739)
        # Checking if sample size 1 is correct
        self.assertEqual(res.treatment_statistics.sample_size, 65)
        # Checking if sample size 2 is correct
        self.assertEqual(res.control_statistics.sample_size, 65)


class SampleSizeTestCases(StatisticsTestCase):
    def test__sample_size__empty_list_numeric(self):
        """ Empty list returns 0. """
        self.assertEqual(statx.sample_size([]), 0)

    def test__sample_size__list_numeric(self):
        """ Result of sample_size() is number of elements of a list. """
        x = [1, 1, 2, 5, 8]
        self.assertEqual(statx.sample_size(x), 5)

    def test__sample_size__nparray_numeric_with_nan(self):
        """ Result of sample_size() is number of elements of numpy array minus number of NaNs. """
        x = np.array([1, 1, np.nan, 2, np.nan, 5, 8])
        self.assertEqual(statx.sample_size(x), 5)

    def test__sample_size__list_categorical(self):
        """ Result of sample_size() is number of elements of a list of categorical data. """
        x = ['1', '1', '3', '2', '6', '5', '8']
        self.assertEqual(statx.sample_size(x), 7)

    def test__sample_size__pdseries_categorical(self):
        """Result of sample_size() is number of elements of a pandas series with categorical data. """
        x = pd.Series(['1', '7', '2', '5', '8', '0'])
        self.assertEqual(statx.sample_size(x), 6)

    def test__sample_size__pdseries_categorical_with_na(self):
        """ Result of sample_size() is number of elements of a pandas series with categorical data including NAs. """
        x = ['1', '1', 'NA', '2', 'NA', '5', '8']
        self.assertEqual(statx.sample_size(x), 5)


class EstimateSampleSizeTestCases(StatisticsTestCase):
    def test__estimate_sample_size__dataframe(self):
        """ Result of estimate_sample_size() is a series of estimated sample sizes. """
        x = pd.DataFrame({'sample_1': [1, 7, 8, 9, 3, 4, 2, 0], 'sample_2': [5, 4, 3, 2, 2, 2, 1, 0]})
        res = statx.estimate_sample_size(x=x, mde=0.01, r=1.0)
        self.assertEqual(int(res['sample_1']), 197405)
        self.assertEqual(int(res['sample_2']), 142130)

    def test__estimate_sample_size__series(self):
        """ Result of estimate_sample_size() is estimated sample size. """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertEqual(int(statx.estimate_sample_size(x=x, mde=0.01, r=1.0)), 197405)

    def test__estimate_sample_size__x_type_error(self):
        """ Method estimate_sample_size raises TypeError since x is a list. """
        x = [1, 7, 8, 9, 3, 4, 2, 0]
        self.assertRaises(TypeError, statx.estimate_sample_size, x=x, mde=0.01, r=1.0)

    def test__estimate_sample_size__r_value_error(self):
        """ Method estimate_sample_size raises ValueError since r is 0. """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertRaises(ValueError, statx.estimate_sample_size, x=x, mde=0.01, r=0.0)

