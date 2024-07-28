"""
    Dummy conftest.py for stresspred.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

from pathlib import Path

import neurokit2 as nk
import pandas as pd
import pytest
from tempbeat.extraction.interval_conversion import peak_time_to_rri


def create_ibi_df():
    """
    Creates a DataFrame containing Inter-Beat Intervals (IBIs).

    Uses a given set of signals, participants, tasks, and rest states.

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
    signals = ["ECG"]
    participants = ["Test1", "Test2", "Test3"]
    tasks = ["MENTAL", "NOISE", "CPT"]
    rest = ["Rest", "Stress"]
    sub_seg_idxs = [1]
    duration = 180
    seg_dfs = []
    random_state = 0
    for signal in signals:
        for participant in participants:
            for task in tasks:
                for rest_val in rest:
                    for sub_seg_idx in sub_seg_idxs:
                        random_state += 1
                        ecg = nk.ecg_simulate(
                            duration=duration, random_state=random_state
                        )
                        signals, info = nk.ecg_process(ecg)
                        ecg = nk.ecg_simulate(duration=duration)
                        _, info = nk.ecg_process(ecg)
                        rri, rri_time = peak_time_to_rri(
                            info["ECG_R_Peaks"] / info["sampling_rate"]
                        )
                        seg_dict = {
                            "Signal": [signal] * len(rri),
                            "Participant": [participant] * len(rri),
                            "Task": [task] * len(rri),
                            "Rest": [rest_val] * len(rri),
                            "SubSegIdx": [sub_seg_idx] * len(rri),
                            "Ibi": rri,
                            "IbiTime": rri_time,
                        }
                        seg_df = pd.DataFrame.from_dict(seg_dict)
                        seg_dfs.append(seg_df)

    ibi_df = pd.concat(seg_dfs)
    return ibi_df


@pytest.fixture(scope="session")
def example_data_paths(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp("data")
    derivatives_dir = Path(data_dir, "derivatives")
    Path(derivatives_dir).mkdir(exist_ok=True)
    stress_class_dir = Path(derivatives_dir, "stress_class")
    Path(stress_class_dir).mkdir(exist_ok=True)
    ibi_df = create_ibi_df()
    ibi_df_path = Path(stress_class_dir, "stress_class_arp_ibi_20220210_fs_500.csv")
    ibi_df.to_csv(str(ibi_df_path), index=False)
    example_data_paths_dict = {
        "data_dir": data_dir,
        "ibi_df": ibi_df_path,
    }
    return example_data_paths_dict
