# The New Hospital Programme Demand Model

<!-- badges: start -->

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![codecov](https://codecov.io/gh/The-Strategy-Unit/nhp_model/branch/main/graph/badge.svg?token=uGmRhc1n9C)](https://codecov.io/gh/The-Strategy-Unit/nhp_model)

<!-- badges: end -->

## Welcome

Welcome to the [New Hospital Programme demand and capacity modelling tool](https://www.strategyunitwm.nhs.uk/new-hospital-programme-demand-model). 

<img width="2000" height="1414" alt="Smarter Hospital Planning" src="https://www.strategyunitwm.nhs.uk/sites/default/files/styles/banner/public/Nationally%20consistent%20approach.png"/>

This repository contains the model code but there are several other repositories which contain useful tools to [explore the data underpinning and set the parameters for the model](https://github.com/The-Strategy-Unit/nhp_inputs), as well as to [explore model outputs](https://github.com/The-Strategy-Unit/nhp_outputs). [An overview of how the different tools interact with each other is available](https://connect.strategyunitwm.nhs.uk/nhp/project_information/project_plan_and_summary/components-overview.html).

The methodology underpinning this model is outlined in this [simple one page explainer](https://connect.strategyunitwm.nhs.uk/nhp_model_explainer/). We have a more technical [project information site](https://connect.strategyunitwm.nhs.uk/nhp/project_information/) which includes further details about the model and the data that the model was built on.

## Running the model

Please note that it is important that the parameters of the model are set with great care and with proper support. It is important also that healthcare system partners are appropriately involved in parameter setting. For a description of the full process and support provision that is necessary to ensure the model functions well please see the [NHS Futures workspace](https://future.nhs.uk/NewHospitalProgrammeDigital/browseFolder?fid=53572528&done=OBJChangesSaved) 

[We are working on providing synthetic data](https://github.com/The-Strategy-Unit/nhp_model/issues/347) so that interested parties can run the model locally to see how it works.

Assuming you have your data in the correct format, store it in the `data` folder. [Further details on the correct formatting for the data to follow](https://github.com/The-Strategy-Unit/nhp_model/issues/419).

The model runs using parameters that are set in a [JSON file](#json-schema).

### Running the model using `uv`

This package is built using [`uv`](https://docs.astral.sh/uv/). If you have `uv` installed, run the model using: `uv run -m nhp.model path/to/params.json -d path/to/data`

### Running the model without `uv`

1. Install the `nhp_model` package using `pip install .`
1. Run the model using: `python -m nhp.model path/to/params.json -d path/to/data`

## Deployment

The model is deployed to Azure Container Registry and GitHub Container Registry on pull requests, tagging the container as `nhp_model:dev`, and on releases its deployed to `nhp_model:v*.*.*` and `nhp_model:latest`.

## JSON Schema

Parameters for the model are set in JSON format; an example can be seen in `src/nhp/model/params/params-sample.json`. As the model develops, requirements for this JSON file change over time. We use [JSON schema](https://json-schema.org/understanding-json-schema/about) to manage changes to the parameters file. From model v3.5 onwards, these are deployed to GitHub pages, following this pattern:
- on merge to `main`, the schema is deployed to `https://the-strategy-unit.github.io/nhp_model/dev/params-schema.json`
- on release of new model version vX.X, the schema is deployed to `https://the-strategy-unit.github.io/nhp_model/vX.X/params-schema.json`
