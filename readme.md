# New Hospitals Demand Model

<!-- badges: start -->

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![codecov](https://codecov.io/gh/The-Strategy-Unit/nhp_model/branch/main/graph/badge.svg?token=uGmRhc1n9C)](https://codecov.io/gh/The-Strategy-Unit/nhp_model)

<!-- badges: end -->

## Welcome

Welcome to the New Hospital Programme demand and capacity modelling tool. The methodology underpinning this model code is given in [the documentation](https://connect.strategyunitwm.nhs.uk/nhp/project_information/), along with a host of other technical information about the model and data that the model was built on.

Please note that it is important that the parameters of the model are set with great care and with proper support. It is important also that healthcare system partners are appropriately involved in parameter setting. For a description of the full process and support provision that is necessary to ensure the model functions well please see the [NHS Futures workspace](https://future.nhs.uk/NewHospitalProgrammeDigital/browseFolder?fid=53572528&done=OBJChangesSaved) 

This repo contains the model code but there are several other repositories which contain useful tools to set the [parameters of and run the model](https://github.com/The-Strategy-Unit/nhp_inputs), as well as to [explore the output of the model](https://github.com/The-Strategy-Unit/nhp_outputs).

## Deployment

The model is deployed to Azure Container Registry on pull requests, tagging the container as `nhp_model:dev`, and on releases its deployed to `nhp_model:v0.*.*` and `nhp_model:latest`.

## JSON Schema

Parameters for the model are set in JSON format; an example can be seen in `queue/sample_params.json`. As the model develops, requirements for this JSON file change over time. We use [JSON schema](https://json-schema.org/understanding-json-schema/about) to manage changes to the parameters file. From model v3.5 onwards, these are deployed to GitHub pages, following this pattern:
- on merge to `main`, the schema is deployed to `https://the-strategy-unit.github.io/nhp_model/dev/params-schema.json`
- on release of new model version vX.X, the schema is deployed to `https://the-strategy-unit.github.io/nhp_model/vX.X/params-schema.json`
