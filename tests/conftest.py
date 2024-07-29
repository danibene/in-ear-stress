"""
    Dummy conftest.py for stresspred.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

from pathlib import Path

import pandas as pd
import pytest

from test_resources.load_resources import get_resource_path

IBI_DF_LINK = "https://raw.githubusercontent.com/danibene/in-ear-stress/5400e91cc1fe896fd43c98bf5e353a5a9493028e/src/test_resources/ibi_df.csv"  # noqa


def get_ibi_df():
    """
    Get an example DataFrame containing Inter-Beat Intervals (IBIs).

    Specifies segments for certain signals, participants, tasks, and rest states.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
            - "Signal" (str): The name of the signal.
            - "Participant" (str): The name of the participant.
            - "Task" (str): The name of the task.
            - "Rest" (str): The rest state (either "Rest" or "Stress").
            - "SubSegIdx" (int): The sub-segment index.
            - "Ibi" (float): The Inter-Beat Interval in seconds.
            - "IbiTime" (float): The time of the Inter-Beat Interval in seconds.
    """
    file_path = get_resource_path("ibi_df.csv")
    if not isinstance(file_path, Path):
        file_path = IBI_DF_LINK
    elif file_path.is_file():
        file_path = IBI_DF_LINK
    return pd.read_csv(file_path)


@pytest.fixture(scope="session")
def example_data_paths(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp("data")
    derivatives_dir = Path(data_dir, "derivatives")
    Path(derivatives_dir).mkdir(exist_ok=True)
    stress_class_dir = Path(derivatives_dir, "stress_class")
    Path(stress_class_dir).mkdir(exist_ok=True)
    ibi_df = get_ibi_df()
    ibi_df_path = Path(stress_class_dir, "stress_class_arp_ibi_20220210_fs_500.csv")
    ibi_df.to_csv(str(ibi_df_path), index=False)
    example_data_paths_dict = {
        "data_dir": data_dir,
        "ibi_df": ibi_df_path,
    }
    return example_data_paths_dict
