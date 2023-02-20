import numpy as np
import pandas as pd

from model.helpers import inrange
from model.model_run import ModelRun


class InpatientEfficiencies:
    def __init__(self, model_run: ModelRun):
        self._model_run = model_run
        self.data = model_run.data
        self.step_counts = model_run.step_counts
        self.strategies = model_run._model._strategies["efficiencies"]
        self._select_single_strategy()
        self._generate_losr_df()

    def _select_single_strategy(self):
        rng = self._model_run.rng
        selected_strategy = (
            self.data["rn"]
            .map(
                self.strategies["strategy"]
                .sample(frac=1, random_state=rng.bit_generator)
                .groupby(level=0)
                .head(1),
                na_action="ignore",
            )
            .fillna("NULL")
            .loc[self.data["rn"]]
            .rename(None)
        )
        self.data.set_index(selected_strategy, inplace=True)

    def _generate_losr_df(self):
        params = self._model_run.params["inpatient_factors"]["los_reduction"]
        run_params = self._model_run.run_params["efficiencies"]["ip"]
        losr = pd.DataFrame.from_dict(params, orient="index")
        losr["losr_f"] = [run_params[i] for i in losr.index]
        self.losr = losr

    def _update(self, i, new):
        mask = ~self.data.loc[i, "bedday_rows"]
        pre_los = self.data.loc[i, "speldur"]
        self.data.loc[i, "speldur"] = new
        change_los = (
            ((self.data.loc[i, "speldur"] - pre_los) * mask)
            .groupby(level=0)
            .sum()
            .astype(int)
        ).to_dict()

        for k in change_los.keys():
            self.step_counts[("efficiencies", k)] = np.array([0, change_los[k]])

        return self

    def losr_all(self):
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
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

        i = losr.index[(losr.type == "all") & (losr.index.isin(data.index))]
        new = rng.binomial(
            data.loc[i, "speldur"], losr.loc[data.loc[i].index, "losr_f"]
        )
        return self._update(i, new)

    def losr_aec(self):
        """
        Length of Stay Reduction: AEC reduction

        Updates the length of stay to 0 for a given percentage of rows.
        """
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

        i = losr.index[(losr.type == "aec") & (losr.index.isin(data.index))]
        new = data.loc[i, "speldur"] * rng.binomial(
            1, losr.loc[data.loc[i].index, "losr_f"]
        )
        return self._update(i, new)

    def losr_preop(self):
        """
        Length of Stay Reduction: Pre-op reduction

        Updates the length of stay to by removing 1 or 2 days for a given percentage of rows
        """
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

        i = losr.index[(losr.type == "pre-op") & (losr.index.isin(data.index))]
        new = data.loc[i, "speldur"] - (
            rng.binomial(1, 1 - losr.loc[data.loc[i].index, "losr_f"])
            * losr.loc[data.loc[i].index, "pre-op_days"]
        )
        return self._update(i, new)

    def losr_bads(self) -> None:
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
        """
        losr = self.losr
        data = self.data
        rng = self._model_run.rng

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

        step_counts_admissions = -(
            ((data.loc[i, "classpat"] == "-1") * ~data.loc[i, "bedday_rows"])
            .groupby(level=0)
            .sum()
            .astype(int)
        )
        step_counts_beddays = (
            (
                (
                    (data.loc[i, "speldur"] * ~data.loc[i, "bedday_rows"])
                    .groupby(level=0)
                    .sum()
                    - (bads_df["speldur"] * ~bads_df["bedday_rows"])
                    .groupby(level=0)
                    .sum()
                    + step_counts_admissions
                )
            )
            .astype(int)
            .to_dict()
        )

        step_counts_admissions = step_counts_admissions.to_dict()
        for k in step_counts_admissions.keys():
            self.step_counts[("efficiencies", k)] = np.array(
                [step_counts_admissions[k], step_counts_beddays[k]]
            )

        return self

    def apply(self):
        self._model_run.data = self.data.reset_index(drop=True)
