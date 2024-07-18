import neurokit2 as nk
import pandas as pd
from tempbeat.extraction.interval_conversion import peak_time_to_rri

from stresspred.data.load_data import StressBioDataLoader


class TestStressBioDataLoader:
    def create_ibi_df(self):
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
        participants = ["Test"]
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

    def test_get_base_feat_df(self):
        """Test function for getting the baseline feature DataFrame."""
        ibi_df = self.create_ibi_df()
        data_loader = StressBioDataLoader()
        base_feat_df = data_loader.get_base_feat_df(
            load_from_file=False, in_data=ibi_df, save_file=False
        )
        index_cols = ["Signal", "Participant", "Task", "Rest", "SubSegIdx"]
        n_unique_segments = ibi_df[index_cols].drop_duplicates().shape[0]
        assert base_feat_df.shape[0] == n_unique_segments
