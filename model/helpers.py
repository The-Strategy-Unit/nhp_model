def inrange(v, lo=0, hi=1):
    """
    Force a value to be in the interval [lo, hi]
    """
    return max(lo, min(hi, v))


def rnorm(rng, lo, hi):
    """
    Create a single random normal value from a 90% confidence interval
    """
    mean = (hi + lo) / 2
    sd = (hi - lo) / 3.289707  # magic number: 2 * qnorm(0.95)
    return rng.normal(mean, sd)
