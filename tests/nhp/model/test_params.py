"""test params-sample"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring

import json

import jsonschema


def test_params_sample():
    # arrange
    with open("queue/params-sample.json") as f:
        params = json.load(f)
    with open("params-schema.json") as f2:
        schema = json.load(f2)

    # act / assert
    jsonschema.validate(params, schema)
