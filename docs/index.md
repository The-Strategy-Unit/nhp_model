# NHP Model Documentation

Welcome to the NHP Model documentation. This project provides modeling capabilities for healthcare activity prediction.

## Features

- Multiple model types (inpatients, outpatients, A&E)
- Support to load data from different sources
- Docker containerization

## Quick Start

Download and install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run `uv sync`. Download data locally, e.g., download a synthetic dataset to `data/synth`. Then, run the model with:

``` bash
uv run python -m nhp.model queue/params-sample.json -d data/synth --type all
```

## API Reference

See the [Model Reference](reference/nhp/model/index.md) for detailed documentation of all classes and functions.