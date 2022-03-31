"""
Helper methods for the model package
"""


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
