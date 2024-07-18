import neurokit2 as nk
import pandas as pd
from tempbeat.extraction.interval_conversion import peak_time_to_rri

from stresspred.data.load_data import StressBioDataLoader


class TestStressBioDataLoader:
    def create_ibi_df(self):
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
        ibi_df = self.create_ibi_df()
        assert ibi_df["Rest"].iloc[0] == "Rest"
        data_loader = StressBioDataLoader()
        data_loader.get_base_feat_df(
            load_from_file=False, in_data=ibi_df, save_file=False
        )
