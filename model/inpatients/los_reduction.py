import numpy as np
import pandas as pd

from model.helpers import inrange


def los_reduction(
    rng: np.random.Generator,
    data: pd.DataFrame,
    los_strategies: pd.DataFrame,
    step_counts: dict,
    params: dict,
    run_params: dict,
) -> None:
    """Perform the los reduction step

    modifies data and step counts in place

    :param rng: the random number generator used in the model
    :type rng: np.random.Generator
    :param data: the data that we are currently working with
    :type data: pd.DataFrame
    :param los_strategies: the strategies available for the admissions
    :type los_strategies: pd.dataFrame
    :param step_counts: the current step counts that contains the changes to our admissions/beddays during each step
    :type step_counts: dictionary
    :param params: the parameters used in this model run
    :type params: dict
    :param run_params: the parameters to use for this model run (see `Model._get_run_params()`)
    :type run_params: dict
    """
    params = params["inpatient_factors"]["los_reduction"]
    # convert the parameters dictionary to a dataframe: each item becomes a row (with the item
    # being the name of the row in the index), and then each sub-item becoming a column
    losr = pd.DataFrame.from_dict(params, orient="index")
    losr["losr_f"] = [
        run_params["inpatient_factors"]["los_reduction"][i] for i in losr.index
    ]

    selected_strategy = (
        data["rn"]
        .map(
            los_strategies["los_reduction_strategy"]
            .sample(frac=1, random_state=rng.bit_generator)
            .groupby(level=0)
            .head(1),
            na_action="ignore",
        )
        .fillna("NULL")
        .loc[data["rn"]]
        .rename(None)
    )

    data.set_index(selected_strategy, inplace=True)
    # run each of the length of stay reduction strategies
    losr_all(data, losr, rng, step_counts)
    losr_aec(data, losr, rng, step_counts)
    losr_preop(data, losr, rng, step_counts)
    losr_bads(data, losr, rng, step_counts)

    data.reset_index(drop=True, inplace=True)


def losr_all(
    data: pd.DataFrame,
    losr: pd.DataFrame,
    rng: np.random.Generator,
    step_counts: dict,
) -> None:
    """Length of Stay Reduction: All

    Reduces all rows length of stay by sampling from a binomial distribution, using the current
    length of stay as the value for n, and the length of stay reduction factor for that strategy
    as the value for p. This will update the los to be a value between 0 and the original los.

    :param data: the DataFrame that we are updating
    :type data: pandas.DataFrame
    :param losr: the Length of Stay rates table created from self._los_reduction()
    :type losr: pandas.DataFrame
    :param rng: a random number generator created for each model iteration
    :type rng: numpy.random.Generator
    :param step_counts: a dictionary containing the changes to measures for this step
    :type step_counts: dict
    """
    i = losr.index[(losr.type == "all") & (losr.index.isin(data.index))]
    pre_los = data.loc[i, "speldur"]
    data.loc[i, "speldur"] = rng.binomial(
        data.loc[i, "speldur"], losr.loc[data.loc[i].index, "losr_f"]
    )
    change_los = (
        (data.loc[i, "speldur"] - pre_los).groupby(level=0).sum().astype(int)
    ).to_dict()

    for k in change_los.keys():
        step_counts[("los_reduction", k)] = {
            "admissions": 0,
            "beddays": change_los[k],
        }


def losr_bads(
    data: pd.DataFrame,
    losr: pd.DataFrame,
    rng: np.random.Generator,
    step_counts: dict,
) -> None:
    """
    Length of Stay Reduction: British Association of Day Surgery

    This will swap rows between elective admissions and daycases into either daycases or
    outpatients, based on the given parameter values. We have a baseline target rate, this is
    the rate in the baseline that rows are of the given target type (i.e. either daycase,
    outpatients, daycase or outpatients). Our interval will alter the target_rate by setting
    some rows which are not the target type to be the target type.

    Rows that are converted to daycase have the patient classification set to 2. Rows that are
    converted to outpatients have a patient classification of -1 (these rows need to be filtered
    out of the inpatients results and added to the outpatients results).

    Rows that are modelled away from elective care have the length of stay fixed to 0 days.

    :param data: the DataFrame that we are updating
    :type data: pandas.DataFrame
    :param losr: the Length of Stay rates table created from self._los_reduction()
    :type losr: pandas.DataFrame
    :param rng: a random number generator created for each model iteration
    :type rng: numpy.random.Generator
    :param step_counts: a dictionary containing the changes to measures for this step
    :type step_counts: dict
    """
    i = losr.type == "bads"
    # create a copy of our data, and join to the losr data
    bads_df = data.merge(
        losr[i][["baseline_target_rate", "op_dc_split", "losr_f"]],
        left_index=True,
        right_index=True,
    )
    # convert the factor value to be the amount we need to adjust the non-target type to to make
    # the target rate equal too the factor value
    factor = (
        (bads_df["losr_f"] - bads_df["baseline_target_rate"])
        / (1 - bads_df["baseline_target_rate"])
    ).apply(inrange)
    # create three values that will sum to 1 - these will be the probabilties of:
    #   - staying where we are  [0.0, u0)
    #   - moving to daycase     [u0,  u1)
    #   - moving to outpatients (u1,  1.0]
    ur1 = bads_df["op_dc_split"] * factor
    ur2 = (1 - bads_df["op_dc_split"]) * factor
    ur0 = 1 - ur1 - ur2
    # we now create a random value for each row in [0, 1]
    rvr = rng.uniform(size=len(factor))
    # we can use this random value to update the patient class column appropriately
    bads_df.loc[
        (rvr >= ur0) & (rvr < ur0 + ur1), "classpat"
    ] = "2"  # row becomes daycase
    bads_df.loc[(rvr >= ur0 + ur1), "classpat"] = "-1"  # row becomes outpatients
    # we now need to apply these changes to the actual data
    i = losr.index[i]
    # make sure we only keep values in losr that exist in data
    i = i[i.isin(data.index)]
    data.loc[i, "classpat"] = bads_df["classpat"].tolist()
    # set the speldur to 0 if we aren't inpatients
    data.loc[i, "speldur"] *= data.loc[i, "classpat"] == "1"

    step_counts_admissions = (
        ((data.loc[i, "classpat"] == "-1").groupby(level=0).sum() * -1)
        .astype(int)
        .to_dict()
    )
    step_counts_beddays = (
        (
            data.loc[i, "speldur"].groupby(level=0).sum()
            - bads_df["speldur"].groupby(level=0).sum()
        )
        .astype(int)
        .to_dict()
    )

    for k in step_counts_admissions.keys():
        step_counts[("los_reduction", k)] = {
            "admissions": step_counts_admissions[k],
            "beddays": step_counts_beddays[k],
        }


def losr_aec(
    data: pd.DataFrame,
    losr: pd.DataFrame,
    rng: np.random.Generator,
    step_counts: dict,
) -> None:
    """
    Length of Stay Reduction: AEC reduction

    Updates the length of stay to 0 for a given percentage of rows.

    :param data: the DataFrame that we are updating
    :type data: pandas.DataFrame
    :param losr: the Length of Stay rates table created from self._los_reduction()
    :type losr: pandas.DataFrame
    :param rng: a random number generator created for each model iteration
    :type rng: numpy.random.Generator
    :param step_counts: a dictionary containing the changes to measures for this step
    :type step_counts: dict
    """
    i = losr.index[(losr.type == "aec") & (losr.index.isin(data.index))]
    pre_los = data.loc[i, "speldur"]
    data.loc[i, "speldur"] *= rng.binomial(1, losr.loc[data.loc[i].index, "losr_f"])
    change_los = (
        (data.loc[i, "speldur"] - pre_los).groupby(level=0).sum().astype(int)
    ).to_dict()

    for k in change_los.keys():
        step_counts[("los_reduction", k)] = {
            "admissions": 0,
            "beddays": change_los[k],
        }


def losr_preop(
    data: pd.DataFrame,
    losr: pd.DataFrame,
    rng: np.random.Generator,
    step_counts: dict,
) -> None:
    """
    Length of Stay Reduction: Pre-op reduction

    Updates the length of stay to by removing 1 or 2 days for a given percentage of rows

    :param data: the DataFrame that we are updating
    :type data: pandas.DataFrame
    :param losr: the Length of Stay rates table created from self._los_reduction()
    :type losr: pandas.DataFrame
    :param rng: a random number generator created for each model iteration
    :type rng: numpy.random.Generator
    :param step_counts: a dictionary containing the changes to measures for this step
    :type step_counts: dict
    """
    i = losr.index[(losr.type == "pre-op") & (losr.index.isin(data.index))]
    pre_los = data.loc[i, "speldur"]
    data.loc[i, "speldur"] -= (
        rng.binomial(1, 1 - losr.loc[data.loc[i].index, "losr_f"])
        * losr.loc[data.loc[i].index, "pre-op_days"]
    )
    change_los = (
        (data.loc[i, "speldur"] - pre_los).groupby(level=0).sum().astype(int)
    ).to_dict()

    for k in change_los.keys():
        step_counts[("los_reduction", k)] = {
            "admissions": 0,
            "beddays": change_los[k],
        }
