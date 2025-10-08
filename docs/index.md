# NHP Model Documentation

Welcome to the NHP Model documentation. This project provides modeling capabilities for healthcare activity prediction.

## Features

- Multiple model types (inpatients, outpatients, A&E)
- Support for loading data from different sources
- Docker containerization

## Quick Start

Download and install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run `uv sync`. Download data locally, e.g., download a synthetic dataset to `data/synth`. Then, run the model with:

``` bash
uv run python -m nhp.model -d data/synth --type all
```

to run the model with the sample parameters.

### Generating Sample Parameters

you can generate sample parameters using the CLI command:

``` bash
python -m nhp.model.params --dataset [dataset] --scenario [scenario] --app-version dev > params.json
```

replacing the values as needed. This will generate a file `params.json` with the sample parameters.

## API Reference

See the [Model Reference](reference/nhp/model/index.md) for detailed documentation of all classes and functions.