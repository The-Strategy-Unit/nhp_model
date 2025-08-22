"""Model Iteration.

Provides a simple class which holds all of the data required for a model iteration
"""

import numpy as np
import pandas as pd

from nhp.model.activity_resampling import ActivityResampling

ModelRunResult = tuple[dict[str, pd.Series], pd.Series | None]


class ModelIteration:
    """Model Iteration.

    Holds all of the information for a model iteration
    """

    def __init__(self, model, model_run: int):
        """Perform an iteration of the model.

        :param model: An instance of a Model object
        :type model: Model
        :param model_run: which model iteration to run
        :type model_run: int
        """
        self.model = model

        self.model_run = model_run
        # if model_run == -1, then use model_run = 0 for run params
        self.run_params = model._get_run_params(max(0, model_run))
        self.rng = np.random.default_rng(self.run_params["seed"])

        self._patch_run_params()

        # data is mutated, so is not a property
        self.data = model.data.copy()
        self.step_counts = None
        self.avoided_activity = pd.DataFrame()

        # run the model
        self._run()

    @property
    def params(self):
        """Get the models parameters."""
        return self.model.params

    def _patch_run_params(self):
        """Patch Run Parameters.

        The run parameters for some items need to be 'patched' so that they include all of the
        fields that are used in that step of the model
        """
        run_params = self.run_params
        for i in ["expat", "repat_local", "repat_nonlocal"]:
            run_params[i]["op"] = {
                g: run_params[i]["op"] for g in ["first", "followup", "procedure"]
            }
            run_params[i]["aae"] = {k: {"Other": v} for k, v in run_params[i]["aae"].items()}

        run_params["baseline_adjustment"]["aae"] = {
            k: {"Other": v} for k, v in run_params["baseline_adjustment"]["aae"].items()
        }

    def _run(self):
        if self.model_run == 0:
            return

        data_ar, step_counts_ar = (
            ActivityResampling(self)
            .demographic_adjustment()
            .birth_adjustment()
            .health_status_adjustment()
            .expat_adjustment()
            .repat_adjustment()
            .waiting_list_adjustment()
            .baseline_adjustment()
            .non_demographic_adjustment()
            .inequalities_adjustment()
            # call apply_resampling last, as this is what actually alters the data
            .apply_resampling()
        )

        data_aa, step_counts_aa = self.model.activity_avoidance(data_ar.copy(), self)
        data_ef, step_counts_ef = self.model.efficiencies(data_aa.copy(), self)

        self.avoided_activity = self.model.calculate_avoided_activity(data_ar, data_aa)

        self.data = data_ef
        self.step_counts = pd.concat(
            [
                self.model.baseline_step_counts,
                step_counts_ar,
                step_counts_aa,
                step_counts_ef,
            ]
        )

    def fix_step_counts(
        self,
        data: pd.DataFrame,
        future: np.ndarray,
        factors: pd.DataFrame,
        term_name: str,
    ) -> pd.DataFrame:
        """Calculate the step counts.

        Calculates the step counts for the current model run, saving back to
        self._model_run.step_counts.

        :param data: the data for the current model run
        :type data: pd.DataFrame
        :param future: the future row counts after running the poisson resampling.
        :type future: np.ndarray
        :param factors: the factors for this current model run
        :type factors: pd.DataFrame
        :param term_name: the name of the interaction term for this step
        :type term_name: str
        :returns:
        :return: the step counts for this step
        :rtype: pd.DataFrame
        """
        before = self.model.get_data_counts(data)
        # convert the paramater values from a dict of 2d numpy arrays to a 3d numpy array
        param_values = np.array(list(factors.to_numpy().transpose()))
        # later on we want to be able to multiply by baseline, we need to have compatible
        # numpy shapes
        # our param values has shape of (x, y). baseline has shape of (z, y).
        # (x, 1, y) will allow (x, y) * (z, y)
        shape = (param_values.shape[0], 1, param_values.shape[1])
        # calculate the simple effect of each parameter, if it was performed in isolation to all
        # other parameters
        param_simple_effects = (param_values - 1).reshape(shape) * before
        # what is the difference left over from the expected changes (model interaction term)
        diff = future - (before + param_simple_effects.sum(axis=0))
        # convert the 3d numpy array back to a pandas dataframe aggregated by the columns we are
        # interested in
        idx = pd.MultiIndex.from_frame(data[["pod", "sitetret"]])
        return pd.concat(
            [
                pd.DataFrame(v.transpose(), columns=self.model.measures, index=idx)
                .groupby(level=idx.names)
                .sum()
                .assign(change_factor=k)
                .reset_index()
                for k, v in {
                    **dict(zip(factors.columns, param_simple_effects)),
                    term_name: diff,
                }.items()
            ]
        )

    def get_aggregate_results(self) -> ModelRunResult:
        """Aggregate the model results.

        Can also be used to aggregate the baseline data by passing in the raw data

        :param model_results: a DataFrame containing the results of a model iteration
        :type model_results: pandas.DataFrame
        :param model_run: the current model run
        :type model_run: int

        :returns: a tuple containing a dictionary of results, and the step counts
        :rtype: tuple[dict[str, pd.Series], pd.Series | None]:
        """
        aggregations = self.model.aggregate(self)

        if not self.avoided_activity.empty:
            avoided_activity_agg = self.model.process_results(self.avoided_activity)
            aggregations["avoided_activity"] = self.model.get_agg(
                avoided_activity_agg, "sex", "age_group"
            )

        return aggregations, self.get_step_counts()

    def get_step_counts(self) -> pd.Series | None:
        """Get the step counts of a model run."""
        if self.step_counts is None:
            return None

        step_counts = (
            self.step_counts.melt(
                [i for i in self.step_counts.columns if i not in self.model.measures],
                var_name="measure",
            )
            .assign(activity_type=self.model.model_type)
            .set_index(
                [
                    "activity_type",
                    "sitetret",
                    "pod",
                    "change_factor",
                    "strategy",
                    "measure",
                ]
            )
            .sort_index()["value"]
        )

        step_counts = self._step_counts_get_type_changes(step_counts)

        return step_counts

    def _step_counts_get_type_changes(self, step_counts) -> pd.Series:
        return pd.concat(
            [
                step_counts,
                self._step_counts_get_type_change_daycase(step_counts),
                self._step_counts_get_type_change_outpatients(step_counts),
                self._step_counts_get_type_change_sdec(step_counts),
            ]
        )  # type: ignore

    def _step_counts_get_type_change_daycase(self, step_counts):
        # get the daycase conversion values
        sc_tc_df = (
            step_counts[
                step_counts.index.isin(
                    ["day_procedures_usually_dc", "day_procedures_occasionally_dc"],
                    level="strategy",
                )
            ]
            .to_frame()
            .reset_index()
        )
        sc_tc_df["pod"] = "ip_elective_daycase"
        sc_tc_df.loc[sc_tc_df["measure"] == "beddays", "value"] = sc_tc_df.loc[
            sc_tc_df["measure"] == "admissions", "value"
        ].tolist()
        return sc_tc_df.groupby(step_counts.index.names)["value"].sum() * -1

    def _step_counts_get_type_change_outpatients(self, step_counts):
        # get the outpatient conversion values
        sc_tc_df = (
            step_counts[
                step_counts.index.isin(
                    ["day_procedures_usually_op", "day_procedures_occasionally_op"],
                    level="strategy",
                )
                & (step_counts.index.get_level_values("measure") == "admissions")
            ]
            .to_frame()
            .reset_index()
        )

        sc_tc_df["activity_type"] = "op"
        sc_tc_df["pod"] = "op_procedure"
        sc_tc_df["measure"] = "attendances"

        return sc_tc_df.groupby(step_counts.index.names)["value"].sum() * -1

    def _step_counts_get_type_change_sdec(self, step_counts):
        # get the sdec conversion values
        sc_tc_df = (
            step_counts[
                step_counts.index.isin(
                    [
                        f"same_day_emergency_care_{i}"
                        for i in ["very_high", "high", "moderate", "low"]
                    ],
                    level="strategy",
                )
                & (step_counts.index.get_level_values("measure") == "admissions")
            ]
            .to_frame()
            .reset_index()
        )

        sc_tc_df["activity_type"] = "aae"
        sc_tc_df["pod"] = "aae_type-05"
        sc_tc_df["measure"] = "arrivals"

        return sc_tc_df.groupby(step_counts.index.names)["value"].sum() * -1

    def get_model_results(self):
        """Get the model results of a model run."""
        return self.data.reset_index(drop=True).drop(columns=["hsagrp"])
