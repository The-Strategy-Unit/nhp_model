"""Inpatient Row Resampling

Methods for handling row resampling"""

import numpy as np
import numpy.typing as npt
import pandas as pd


def expat_adjustment(data: pd.DataFrame, run_params: dict) -> npt.NDArray:
    expat_params = run_params["expat"]["ip"]
    join_cols = ["admigroup", "tretspef"]
    return (
        data.merge(
            pd.DataFrame(
                [
                    {"admigroup": k1, "tretspef": k2, "value": v2}
                    for k1, v1 in expat_params.items()
                    for k2, v2 in v1.items()
                ]
            ).set_index(join_cols),
            how="left",
            left_on=join_cols,
            right_index=True,
        )["value"]
        .fillna(1)
        .to_numpy()
    )


def repat_adjustment(data: pd.DataFrame, run_params: dict) -> npt.NDArray:
    repat_local_params = run_params["repat_local"]["ip"]
    repat_nonlocal_params = run_params["repat_nonlocal"]["ip"]
    join_cols = ["admigroup", "tretspef", "is_main_icb"]
    return (
        data.merge(
            pd.DataFrame(
                [
                    {"admigroup": k1, "tretspef": k2, "is_main_icb": icb, "value": v2}
                    for (k0, icb) in [
                        (repat_local_params, True),
                        (repat_nonlocal_params, False),
                    ]
                    for k1, v1 in k0.items()
                    for k2, v2 in v1.items()
                ]
            ).set_index(join_cols),
            how="left",
            left_on=join_cols,
            right_index=True,
        )["value"]
        .fillna(1)
        .to_numpy()
    )


def baseline_adjustment(data: pd.DataFrame, run_params: dict) -> npt.NDArray:
    """Create a series of factors for baseline adjustment.

    A value of 1 will indicate that we want to sample this row at the baseline rate. A value
    less that 1 will indicate we want to sample that row less often that in the baseline, and
    a value greater than 1 will indicate that we want to sample that row more often than in the
    baseline

    :param data: the DataFrame that we are updating
    :type data: pandas.DataFrame
    :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
    :type run_params: dict

    :returns: a series of floats indicating how often we want to sample that row
    :rtype: pandas.Series
    """
    params = pd.concat(
        {
            k: pd.Series(v, name="baseline_adjustment")
            for k, v in run_params["baseline_adjustment"]["ip"].items()
        }
    )
    return data.merge(
        params, how="left", left_on=["admigroup", "tretspef"], right_index=True
    )["baseline_adjustment"].fillna(1).to_numpy()


def waiting_list_adjustment(data: pd.DataFrame, run_params: dict) -> npt.NDArray:
    """Create a series of factors for waiting list adjustment.

    A value of 1 will indicate that we want to sample this row at the baseline rate. A value
    less that 1 will indicate we want to sample that row less often that in the baseline, and
    a value greater than 1 will indicate that we want to sample that row more often than in the
    baseline

    :param data: the DataFrame that we are updating
    :type data: pandas.DataFrame

    :returns: a series of floats indicating how often we want to sample that row
    :rtype: pandas.Series
    """
    tretspef_n = data.groupby("tretspef").size()
    wla_param = pd.Series(run_params["waiting_list_adjustment"]["ip"])
    wla = (wla_param / tretspef_n).fillna(0) + 1
    # waiting list adjustment values
    # create a series of 1's the length of our data: make sure these are floats, not ints
    wlav = np.ones_like(data.index).astype(float)
    # find the rows which are elective wait list admissions
    i = list(data.admimeth == "11")
    # update the series for these rows with the waiting list adjustments for that specialty
    wlav[i] = wla[data[i].tretspef]
    # return the waiting list adjustment factor series
    return wlav


def non_demographic_adjustment(data: pd.DataFrame, run_params: dict) -> npt.NDArray:
    """Create a series of factors for non-demographic adjustment.

    :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
    :type run_params: dict

    :returns: a series of floats indicating how often we want to sample that row)
    :rtype: pandas.Series
    """
    ndp = (
        pd.DataFrame.from_dict(run_params["non-demographic_adjustment"], orient="index")
        .reset_index()
        .rename(columns={"index": "admigroup"})
        .melt(["admigroup"], var_name="age_group")
        .set_index(["age_group", "admigroup"])["value"]
    )

    return (
        data[["age_group", "admigroup"]]
        .join(ndp, on=["age_group", "admigroup"])["value"]
        .to_numpy()
    )

def admission_avoidance(
    rng: np.random.Generator,
    data: pd.DataFrame,
    strategies: pd.DataFrame,
    run_params: dict,
) -> dict:
    p = pd.Series(run_params["inpatient_factors"]["admission_avoidance"], name="aaf")

    return {
        ("admission_avoidance", k): data["rn"].map(v, na_action="ignore").fillna(1).to_numpy()
        for k, v in strategies.merge(
            p, left_on="admission_avoidance_strategy", right_index=True
        )
        .assign(aaf=lambda x: 1 - rng.binomial(1, x["sample_rate"]) * (1 - x["aaf"]))
        .groupby("admission_avoidance_strategy")["aaf"]
    }

def apply_resampling(rng: np.random.Generator, data: pd.DataFrame, factors: dict):
    # create a value for each row that is a complex number where
    #   - the real part is a count of how many admissions that row represents
    #   - the imaginary part is how many beddays that row represents
    # we zero out the "beddays" rows, as these shouldn't count towards our rows counts
    n = (1 + (1 + data["speldur"]) * 1j) * (~data["bedday_rows"]).astype(float)
    # sum the baseline values
    sb = s = n.sum()

    cf = {}

    r = 1
    for k, v in factors.items():
        tn = n * v
        ts = tn.sum()
        cf[k] = ts - s
        n, s = tn, ts
        r *= v

    admissions = rng.poisson(r)

    data_n = data.loc[data.index.repeat(admissions)].reset_index(drop=True)

    # calculate the slack
    n = (1 + (1 + data_n["speldur"]) * 1j) * (~data_n["bedday_rows"]).astype(float)
    sa = n.sum()

    a_slack = 1 + (sa - s).real / (s - sb).real
    b_slack = 1 + (sa - s).imag / (s - sb).imag

    cf = {
        ("baseline", "-"): {
            "admissions": sb.real,
            "beddays": sb.imag
        },
        **{
            k: {
                "admissions": v.real * a_slack,
                "beddays": v.imag * b_slack
            }
            for k, v in cf.items()
        }
    }

    return (data_n, cf)