# TODO: This module is deprecated

import sys
import unittest

import pandas as pd

from expan.core.binning import *

#---------- test util function ------------#
def toBinRepresentation(bins):
    """
    Get the bin representations of the given bins
    :param bins: a list of Bin objects.
    :return: a lista list of NumericalRepresenation or CategoricalRepresenation objects
    """
    reprs = []
    for bin in bins:
        reprs.append(bin.representation)
    return reprs


#---------- test classes ------------#
class BinningTestCase(unittest.TestCase):
    """
    Defines the setUp() and tearDown() functions for the statistics test cases.
    """
    def setUp(self):
        np.random.seed(41)
        self.maxDiff = None

    def tearDown(self):
        pass

    def assertCollectionEqual(self, source, expected):
        is_python3 = sys.version_info[0] == 3
        if is_python3:
            return self.assertCountEqual(source, expected)
        else:
            return self.assertItemsEqual(source, expected)



#---------- Numerical binning tests ------------#
class CreateNumericalBinsTestCase(BinningTestCase):
    """
    Test cases for creating numerical bins.
    """
    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            create_bins(None, 10)
        with self.assertRaises(ValueError):
            create_bins(list(range(100)), 0)

    def test_create_regular(self):
        data = np.arange(1000)
        nbins = 10
        bins = create_bins(data, nbins)
        self.assertEqual(len(bins), nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 100, True, False),
            NumericalRepresentation(100, 200, True, False),
            NumericalRepresentation(200, 300, True, False),
            NumericalRepresentation(300, 400, True, False),
            NumericalRepresentation(400, 500, True, False),
            NumericalRepresentation(500, 600, True, False),
            NumericalRepresentation(600, 700, True, False),
            NumericalRepresentation(700, 800, True, False),
            NumericalRepresentation(800, 900, True, False),
            NumericalRepresentation(900, 999, True, True)
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_create_nan(self):
        data = np.arange(1002.)
        data[-2] = np.nan
        data[-1] = np.nan
        nbins = 11
        bins = create_bins(data, nbins)
