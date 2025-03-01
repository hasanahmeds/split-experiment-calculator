import logging
import warnings

import numpy as np
import pandas as pd
import copy

import expan.core.early_stopping as es
import expan.core.statistics as statx
import expan.core.correction as correction
from expan.core.statistical_test import *
from expan.core.results import StatisticalTestResult, MultipleTestSuiteResult, CombinedTestStatistics

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

DEFAULT_OUTLIER_QUANTILE = 0.99

class Experiment(object):
    """ Class which adds the analysis functions to experimental data. """
    def __init__(self, metadata):
        """
        Constructor of the experiment object.
        
        :param metadata: additional information about the experiment. (e.g. primary KPI, source, etc)
        :type  metadata: dict
        """
        self.metadata     = metadata
        self.worker_table = {
            'fixed_horizon': statx.make_delta,
            'group_sequential': es.make_group_sequential,
            'bayes_factor': es.make_bayes_factor,
            'bayes_precision': es.make_bayes_precision
        }

    def __str__(self):
        return 'Performing "{:s}" experiment.'.format(self.metadata['experiment'])


    def analyze_statistical_test(self, test, test_method='fixed_horizon', include_data=False, **worker_args):
        """ Runs delta analysis on one statistical test and returns statistical results.
        
        :param test: a statistical test to run
        :type  test: StatisticalTest
        :param test_method: analysis method to perform. 
                           It can be 'fixed_horizon', 'group_sequential', 'bayes_factor' or 'bayes_precision'.
        :type  test_method: str
        :param include_data: True if test results should include data, False - if no data should be included
        :type  include_data: bool
        :param worker_args: additional arguments for the analysis method

        :return: statistical result of the test
        :rtype: StatisticalTestResult
        """
        if not isinstance(test, StatisticalTest):
            raise TypeError("Statistical test should be of type StatisticalTest.")

        if 'entity' not in test.data.columns:
            raise KeyError("There is no 'entity' column in the data.")
        if test.variants.variant_column_name not in test.data.columns:
            raise KeyError("There is no '{}' column in the data.".format(test.variants.variant_column_name))

        for feature in test.features:
            if feature.column_name not in test.data.columns:
                raise KeyError("Feature name '{}' does not exist in the data.".format(feature.column_name))

        if type(test.kpi) is KPI and (test.kpi.name not in test.data.columns):
            raise KeyError("There is no column of name '{}' in the data.".format(test.kpi.name))
        if type(test.kpi) is DerivedKPI:
            if type(test.kpi.numerator) is not str or test.kpi.numerator not in test.data.columns:
                raise KeyError("Numerator '{}' of the derived KPI does not exist in the data.".format(test.kpi.numerator))
            if type(test.kpi.denominator) is not str or test.kpi.denominator not in test.data.columns:
                raise KeyError("Denominator '{}' of the derived KPI does not exist in the data.".format(test.kpi.denominator))
            test.kpi.make_derived_kpi(test.data)

        logger.info("One analysis with kpi '{}', control variant '{}', treatment variant '{}' and features [{}] "
                    "has just started".format(test.kpi, test.variants.control_name,
                                              test.variants.treatment_name,
                                              [(feature.column_name, feature.column_value) for feature in test.features]))

        if test_method not in self.worker_table:
            raise NotImplementedError("Test method '{}' is not implemented.".format(test_method))
        worker = self.worker_table[test_method](**worker_args)

        # create test result object with empty result first
        test_result = StatisticalTestResult(test, None)

        data_for_analysis = test.data
        for feature in test.features:
            data_for_analysis = feature.apply_to_data(data_for_analysis)

        if not self._is_valid_for_analysis(data_for_analysis, test):
            # Note that this does not check that there are enough
            # non-NaN and non=Inf datapoints. See below for a check
            # of that:
            logger.warning("Data is not valid for the analysis!")
            return test_result
        if data_for_analysis.entity.duplicated().any():
            raise ValueError('Entities in data should be unique.')

        # get control and treatment values for the kpi
        control          = test.variants.get_variant(data_for_analysis, test.variants.control_name)[test.kpi.name]
        logger.info("Control group size: {}".format(control.shape[0]))
        control_denominators   = self._get_denominators(data_for_analysis, test, test.variants.control_name)
        control_numerators   = control * control_denominators

        treatment        = test.variants.get_variant(data_for_analysis, test.variants.treatment_name)[test.kpi.name]
        logger.info("Treatment group size: {}".format(treatment.shape[0]))
        treatment_denominators = self._get_denominators(data_for_analysis, test, test.variants.treatment_name)
        treatment_numerators   = treatment * treatment_denominators

        number_of_finite_controls   = np.sum(np.isfinite( control_numerators   / control_denominators   ))
        number_of_finite_treatments = np.sum(np.isfinite( treatment_numerators / treatment_denominators ))

        if number_of_finite_controls < 2 or number_of_finite_treatments < 2: return test_result

        # run the test method
        test_statistics = worker(x=treatment_numerators, y=control_numerators, x_denominators = treatment_denominators, y_denominators = control_denominators)
        test_result.result = test_statistics

        # remove data from the result test metadata
        if not include_data:
            del test.data
        return test_result


    def analyze_statistical_test_suite(self, test_suite, test_method='fixed_horizon', **worker_args):
        """ Runs delta analysis on a set of tests and returns statistical results for each statistical test in the suite.
        
        :param test_suite: a suite of statistical test to run
        :type  test_suite: StatisticalTestSuite
        :param test_method: analysis method to perform. 
                           It can be 'fixed_horizon', 'group_sequential', 'bayes_factor' or 'bayes_precision'.
        :type  test_method: str
        :param worker_args: additional arguments for the analysis method (see signatures of corresponding methods)

        :return: statistical result of the test suite
        :rtype: MultipleTestSuiteResult
        """
        if not isinstance(test_suite, StatisticalTestSuite):
            raise TypeError("Test suite should be of type StatisticalTestSuite.")

        if test_method not in ['fixed_horizon', 'group_sequential']:
            test_suite.correction_method = CorrectionMethod.NONE
        requires_correction = test_suite.correction_method is not CorrectionMethod.NONE

        # look up table for correction method
        correction_table = {
            CorrectionMethod.BONFERRONI: correction.bonferroni,
            CorrectionMethod.BH: correction.benjamini_hochberg
        }

        logger.info("Statistical test suite analysis with {} tests, testmethod {}, correction method {} "
                    "has just started".format(len(test_suite.tests), test_method, test_suite.correction_method))

        # test_suite_result hold statistical results from all statistical tests
        test_suite_result = MultipleTestSuiteResult([], test_suite.correction_method)
        for test in test_suite.tests:
            original_analysis = self.analyze_statistical_test(test, test_method, True, **worker_args)
            # If the statistical power is -1 *or* if the results are None, then we don't include the analysis
