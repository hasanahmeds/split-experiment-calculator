import os
import logging
import pickle
import sys
import tempfile
from os.path import dirname, join, realpath

import pandas as pd
import numpy as np
from pystan import StanModel
from scipy.stats import gaussian_kde, norm, cauchy

import expan.core.statistics as statx
from expan.core.util import drop_nan
from expan.core.results import BaseTestStatistics, SampleStatistics, EarlyStoppingTestStatistics

__location__ = realpath(join(os.getcwd(), dirname(__file__)))
logger = logging.getLogger(__name__)

cache_sampling_results = False
sampling_results = {}  # memorized sampling results


def obrien_fleming(information_fraction, alpha=0.05):
    """ Calculate an approximation of the O'Brien-Fleming alpha spending function.

    :param information_fraction: share of the information  amount at the point of evaluation, 
                                 e.g. the share of the maximum sample size
    :type  information_fraction: float
    :param alpha: type-I error rate
    :type  alpha: float

    :return: redistributed alpha value at the time point with the given information fraction
    :rtype:  float
    """
    return (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(information_fraction))) * 2


def make_group_sequential(spending_function='obrien_fleming', estimated_sample_size=None, alpha=0.05, cap=8):
    """ A closure to the group_sequential function. """
    def go(x, y, x_denominators=1, y_denominators=1):

        # these next too lines are wrong, but they are bug-compatible with v0.6.13 !
        x = x / np.nanmean(x_denominators)
        y = y / np.nanmean(y_denominators)

        return group_sequential(x, y, spending_function, estimated_sample_size, alpha, cap)
    return go


def group_sequential(x, y, spending_function='obrien_fleming', estimated_sample_size=None, alpha=0.05, cap=8):
    """ Group sequential method to determine whether to stop early.

    :param x: sample of a treatment group
    :type  x: pd.Series or array-like
    :param y: sample of a control group
    :type  y: pd.Series or array-like
    :param spending_function: name of the alpha spending function, currently supports only 'obrien_fleming'.
    :type  spending_function: str
    :param estimated_sample_size: sample size to be achieved towards the end of experiment
    :type  estimated_sample_size: int
    :param alpha: type-I error rate
    :type  alpha: float
    :param cap: upper bound of the adapted z-score
    :type  cap: int

    :return: results of type EarlyStoppingTestStatistics
    :rtype:  EarlyStoppingTestStatistics
    """
    # Checking if data was provided and it has correct format
    if x is None or y is None:
        raise ValueError('Please provide two non-empty samples.')
    if not isinstance(x, pd.Series) and not isinstance(x, np.ndarray) and not isinstance(x, list):
        raise TypeError('Please provide samples of type Series or list.')
    if type(x) != type(y):
        raise TypeError('Please provide samples of the same type.')

    logger.info("Started running group sequential early stopping; spending function is {}, size of treatment is {} "
                "and size of control is {}".format(spending_function, len(x), len(y)))

    # Coercing missing values to right format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)

    n_x = statx.sample_size(_x)
    n_y = statx.sample_size(_y)

    if not estimated_sample_size:
        information_fraction = 1.0
    else:
        information_fraction = min(1.0, (n_x + n_y) / estimated_sample_size)

    # alpha spending function
    if spending_function in ('obrien_fleming'):
        func = eval(spending_function)
    else:
        raise NotImplementedError
    alpha_new = func(information_fraction, alpha=alpha)

    # calculate the z-score bound
    bound = norm.ppf(1 - alpha_new / 2)
    # replace potential inf with an upper bound
    if bound == np.inf:
        bound = cap

    mu_x = np.nanmean(_x)
    mu_y = np.nanmean(_y)
    sigma_x = np.nanstd(_x)
    sigma_y = np.nanstd(_y)
    z = (mu_x - mu_y) / np.sqrt(sigma_x ** 2 / n_x + sigma_y ** 2 / n_y)

    if z > bound or z < -bound:
        stop = True
    else:
        stop = False

    interval = statx.normal_difference(mu_x, sigma_x, n_x, mu_y, sigma_y, n_y,
                                       [alpha_new * 100 / 2, 100 - alpha_new * 100 / 2])

    treatment_statistics = SampleStatistics(int(n_x), float(np.nanmean(_x)), float(np.nanvar(_x)))
    control_statistics   = SampleStatistics(int(n_y), float(np.nanmean(_y)), float(np.nanvar(_y)))
    variant_statistics   = BaseTestStatistics(control_statistics, treatment_statistics)
    p_value              = statx.compute_p_value_from_samples(_x, _y)
    statistical_power    = statx.compute_statistical_power_from_samples(_x, _y, alpha)

    logger.info("Finished running group sequential early stopping; spending function is {}, size of treatment is {} "
                "and size of control is {}".format(spending_function, len(x), len(y)))
    return EarlyStoppingTestStatistics(variant_statistics.control_statistics,
                                       variant_statistics.treatment_statistics,
                                       float(mu_x - mu_y), interval, p_value, statistical_power, stop)


def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    """ Computes highest density interval from a sample of representative values, 
    estimated as the shortest credible interval.
    Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95).
    http://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    
    :param posterior_samples: sample of data points from posterior distribution of some parameter
    :type  posterior_samples: array-like
    :param credible_mass: the range of credible interval. 0.95 means 95% represents credible interval.
    :type  credible_mass: float

    :return: corresponding lower and upper bound for the credible interval
    :rtype:  tuple[float]
    """

    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0] * nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]

    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth)) + ciIdxInc]
    return (HDImin, HDImax)


def _bayes_sampling(x, y, distribution='normal', num_iters=25000, inference="sampling"):
    """ Helper function for bayesian sampling.

    :param x: sample of a treatment group
    :type  x: pd.Series or list (array-like)
    :param y: sample of a control group
    :type  y: pd.Series or list (array-like)
    :param distribution: name of the KPI distribution model, which assumes a Stan model file with the same name exists
    :type  distribution: str
    :param num_iters: number of iterations of sampling
    :type  num_iters: int
    :param inference: 'sampling' for MCMC sampling method or 'variational' for variational inference
    :type  inference: str

    :return: the posterior samples, sample size of x, sample size of y, absolute mean of x, absolute mean of y
    :rtype:  tuple[array-like, array-like, array-like, float, float]
    """
    # Checking if data was provided and it has correct format
    if x is None or y is None:
        raise ValueError('Please provide two non-empty samples.')
    if not isinstance(x, pd.Series) and not isinstance(x, np.ndarray) and not isinstance(x, list):
        raise TypeError('Please provide samples of type Series or list.')
    if type(x) != type(y):
        raise TypeError('Please provide samples of the same type.')

    logger.info("Started running bayesian inference with {} procedure, treatment group of size {}, "
                "control group of size {}, {} distribution.".format(inference, len(x), len(y), distribution, inference))

    # Coercing missing values to right format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)
    _x = drop_nan(_x)
    _y = drop_nan(_y)

    key = (str(_x), str(_y), num_iters, inference)

    if cache_sampling_results and key in sampling_results:
        return sampling_results[key]

    mu_x = np.nanmean(_x)
    mu_y = np.nanmean(_y)
