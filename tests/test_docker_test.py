"""test docker test"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, patch

import config
from docker_test import main, progress_callback

config.STORAGE_ACCOUNT = "sa"
config.APP_VERSION = "dev"
config.DATA_VERSION = "dev"


def test_progress_callback():
    assert not progress_callback(1)(2)


def test_main(mocker):
    # arrange
    args = Mock()
    args.params_file = "params.json"

    mocker.patch("argparse.ArgumentParser", return_value=args)
    args.parse_args.return_value = args

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]
    params["dict"] = {"a": 1}

    lp_m = mocker.patch("docker_test.load_params", return_value=params)
    ru_m = mocker.patch("docker_test.run_all", return_value="results.json")

    # act
    main()

    # assert
    assert args.add_argument.call_args_list == [
        call(
            "params_file",
            nargs="?",
            default="sample_params.json",
            help="Name of the parameters file stored in Azure",
        )
    ]
    args.parse_args.assert_called_once()

    lp_m.assert_called_once_with("queue/params.json")
    ru_m.assert_called_once_with(params, "data", progress_callback)


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import docker_test as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("docker_test.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
