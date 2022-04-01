"""
Helper methods for the model package
"""

import numpy as np
import pandas as pd


def inrange(value, low=0, high=1):
    """
    Force a value to be in the interval [lo, hi]
    """
    return max(low, min(high, value))


def rnorm(rng, low, high):
    """
    Create a single random normal value from a 90% confidence interval
    """
    mean = (high + low) / 2
    stdev = (high - low) / 3.289707  # magic number: 2 * qnorm(0.95)
    return rng.normal(mean, stdev)


def np_encoder(obj):
    """
    handle encoding of numpy values

    (source: https://stackoverflow.com/a/65151218/4636789)
    """
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def age_groups(age):
    """
    cut age into groups
    """
    return pd.cut(
        age,
        [0, 5, 15, 35, 50, 65, 85, 1000],
        False,
        ["0-4", "5-14", "15-34", "35-49", "50-64", "65-84", "85+"],
    ).astype(str)
