import os
import subprocess
import sys

import pytest

# Define activity types
ACTIVITY_TYPES = ["ip", "op", "aae"]


@pytest.mark.parametrize("activity_type", ACTIVITY_TYPES)
def test_activity_execution(activity_type, test_data_dir):
    """Test that each activity type executes successfully."""

    # Skip test in CI environment
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping test in CI environment as it requires test data")

    # Expand the home directory if path contains tilde
    expanded_dir = os.path.abspath(os.path.expanduser(test_data_dir))

    # Verify the test data directory exists
    if not os.path.isdir(expanded_dir):
        pytest.skip(f"Test data directory does not exist: {expanded_dir}")

    # Determine the params file path
    # Go up one level from the test file to the project root, then to queue/params-sample.json
    params_file = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queue/params-sample.json"
        )
    )

    # Verify the params file exists
    if not os.path.isfile(params_file):
        pytest.skip(f"Params file does not exist: {params_file}")

    command = [
        "uv",
        "run",
        "python",
        "-m",
        "nhp.model",
        params_file,
        "-d",
        expanded_dir,
        "--type",
        activity_type,
    ]

    try:
        result = subprocess.run(
            args=command,
            capture_output=True,
            text=True,
            check=False,
        )

        # Print debugging information on failure
        if result.returncode != 0:
            print(f"Command failed: {' '.join(command)}")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")

        assert result.returncode == 0, f"Activity {activity_type} failed: {result.stderr}"

    except Exception as e:
        pytest.fail(f"Error executing command: {e!s}")
