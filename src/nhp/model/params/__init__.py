"""Module for working with model parameter files."""

import json

from importlib_resources import files

from nhp.model import params as params_module


def validate_params(params: dict) -> None:
    """Validate model parameters.

    Args:
        params: The model parameters to validate.

    Raises:
        jsonschema.ValidationError: If the parameters are not valid.
    """
    # lazy load for test collection performance
    import jsonschema  # noqa: PLC0415

    with (
        files(params_module)
        .joinpath("params-schema.json")
        .open("r", encoding="UTF-8") as schema_file
    ):
        schema = json.load(schema_file)

    jsonschema.validate(instance=params, schema=schema)


def load_params(filename: str) -> dict:
    """Load a params file.

    Args:
        filename: The full name of the file that we wish to load.

    Raises:
        jsonschema.ValidationError: If the parameters are not valid.

    Returns:
        The model parameters.
    """
    with open(filename, "r", encoding="UTF-8") as prf:
        params = json.load(prf)

    validate_params(params)

    return params


def load_sample_params(**kwargs) -> dict:
    """Load a sample params file.

    Args:
        **kwargs: Any parameters to override in the sample params.

    Raises:
        jsonschema.ValidationError: If the parameters are not valid.

    Returns:
        The model parameters.
    """
    with files(params_module).joinpath("params-sample.json").open("r", encoding="UTF-8") as prf:
        params = json.load(prf)

    params.update(kwargs)

    validate_params(params)

    return params
