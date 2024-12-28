from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from tempbeat.extraction.interval_conversion import peak_time_to_rri


def simulate_ecg_signal(sampling_rate=250, duration=180, random_state=0, rest="Rest"):
    rng = np.random.default_rng(random_state)
    if rest == "Rest":
        max_hr = 100
        min_hr = 40
    else:
        max_hr = 200
        min_hr = 60
    heart_rate = rng.integers(min_hr, max_hr)
    return nk.ecg_simulate(
        duration=duration,
        random_state=random_state,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
    )


def create_noisy_rri_signal(
    clean_ecg,
    sampling_rate=250,
    rng=None,
    signal_name="ECG",
    clean_signal_name="ECG",
    rest="Rest",
):
    if rng is None:
        rng = np.random.default_rng()
    if signal_name == clean_signal_name:
        ecg = clean_ecg
        noise_amplitude = 0
    else:
        if rest == "Rest":
            # Assuming fewer artifacts when participants at rest
            noise_amplitude = rng.integers(5, 20) / 100
        else:
            noise_amplitude = rng.integers(10, 75) / 100

        ecg = nk.signal_distort(
            signal=clean_ecg,
            sampling_rate=sampling_rate,
            noise_amplitude=noise_amplitude,
            noise_frequency=10,
            random_state=rng.integers(0, 10000),
        )

    _, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    rri, rri_time = peak_time_to_rri(info["ECG_R_Peaks"] / info["sampling_rate"])

    if noise_amplitude > 0.5:
        # Simulate missing R-R intervals for very noisy signals
        n_intervals_to_remove = rng.integers(0, 10)
        start_index = rng.integers(0, len(rri) - n_intervals_to_remove)
        rri = np.delete(rri, range(start_index, start_index + n_intervals_to_remove))
        rri_time = np.delete(
            rri_time, range(start_index, start_index + n_intervals_to_remove)
        )

    return rri, rri_time


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
    participants = ["FakeParticipant" + str(i + 1) for i in np.arange(15)]
    tasks = ["MENTAL", "NOISE", "CPT"]
    rest = ["Rest", "Stress"]
    sub_seg_idxs = [1]
    sampling_rate = 100
    seg_dfs = []
    random_state = 0
    signal_names = ["ECG", "IEML", "IEMR"]

    for participant in participants:
        print(participant)
        for task in tasks:
            for rest_val in rest:
                for sub_seg_idx in sub_seg_idxs:
                    random_state += 1
                    rng = np.random.default_rng(random_state)
                    clean_ecg = simulate_ecg_signal(
                        sampling_rate=sampling_rate,
                        random_state=random_state,
                        rest=rest_val,
                    )

                    for signal_name in signal_names:
                        rri, rri_time = create_noisy_rri_signal(
                            clean_ecg=clean_ecg,
                            sampling_rate=sampling_rate,
                            rng=rng,
                            signal_name=signal_name,
                            rest=rest_val,
                        )
                        seg_dict = {
                            "Signal": [signal_name] * len(rri),
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


if __name__ == "__main__":
    ibi_df = create_ibi_df()
    ibi_df.to_csv(Path(__file__).parent / "ibi_df.csv", index=False)
