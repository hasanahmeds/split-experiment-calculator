""" NB: This module is deprecated. """

import logging
import warnings
from heapq import heapify, heappush, heappop

import numpy as np

from expan.core.util import is_nan

logger = logging.getLogger(__name__)


#------- bin representations -------#
class Bin(object):
    # this is a necessary hack for the buggy implementation of assertItemsEqual in python2
    # see https://stackoverflow.com/a/29690198
    # note that if we only use python3, assertCountEqual in python3 solves this problem
    __hash__ = None

    def __init__(self, bin_type, *repr_args):
        """
        Constructor for a bin object.
        :param id: identifier (e.g. bin number) of the bin
        :param bin_type: "numerical" or "categorical"
        :param repr_args: arguments to represent this bin. 
        args for numerical bin includes lower, upper, lower_closed, upper_closed
        args for categorical bin includes a list of categories for this bin.
        """
        if bin_type == "numerical" and len(repr_args) != 4:
            raise ValueError("args for numerical bin are lower, upper, lower_closed, upper_closed.")
        if bin_type == "categorical" and len(repr_args) != 1 and type(repr_args[0]) is not list:
            raise ValueError("args for categorical bin is a list of categorical values for this bin.")
        self.bin_type = bin_type

        if bin_type == "numerical":
            self.representation = NumericalRepresentation(*repr_args)
        elif bin_type == "categorical":
            self.representation = CategoricalRepresentation(*repr_args)

    def __repr__(self):
        return "\nbin: " + str(self.representation)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, data, feature):
        """
        Apply the bin to data.
        :param data: pandas data frame
        :param feature: feature name on which this bin is defined
        :return: subset of input dataframe which belongs to this bin
        """
        return self.representation.apply_to_data(data, feature)


class NumericalRepresentation(object):
    # this is a necessary hack for the buggy implementation of assertItemsEqual in python2
    # see https://stackoverflow.com/a/29690198
    # note that if we only use python3, assertCountEqual in python3 solves this problem
    __hash__ = None

    def __init__(self, lower, upper, lower_closed, upper_closed):
        """
        Constructor for representation of a numerical bin.
        :param upper: upper bound of the bin (exclusive)
        :param lower: lower bound of the bin (inclusive)
        :param lower_closed: boolean indicator whether lower bound is closed
        :param upper_closed: boolean indicator whether upper bound is closed
        """
        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed

    def __repr__(self):
        repr = ""
        if self.lower_closed:
            repr += "["
        else:
            repr += "("

        repr += str(self.lower) + ", " + str(self.upper)

        if self.upper_closed:
            repr += "]"
        else:
            repr += ")"
        return repr

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_to_data(self, data, feature):
        """
        Apply the bin to data.
        :param data: pandas data frame
        :param feature: feature name on which this bin is defined
        :return: subset of input dataframe which belongs to this bin
        """
        data_feature_column = data[feature]

        # if either bound is nan, only nans exist in the bin.
        if np.isnan(self.lower) or np.isnan(self.upper):
            return data[np.isnan(data_feature_column)]

        if self.lower_closed:
            filter_lower = (data_feature_column >= self.lower)
        else:
            filter_lower = (data_feature_column > self.lower)
        if self.upper_closed:
            filter_upper = (data_feature_column <= self.upper)
        else:
            filter_upper = (data_feature_column < self.upper)
