import pandas as pd

from stresspred.data.load_data import AudaceDataLoader, StressBioDataLoader


class TestStressBioDataLoader:
    @staticmethod
    def test_get_base_feat_df(example_data_paths):
        """Test function for getting the baseline feature DataFrame."""
        ibi_df = pd.read_csv(str(example_data_paths["ibi_df"]))
        data_loader = StressBioDataLoader()
        base_feat_df = data_loader.get_base_feat_df(
            load_from_file=False, in_data=ibi_df, save_file=False
        )
        index_cols = ["Signal", "Participant", "Task", "Rest", "SubSegIdx"]
        n_unique_segments = ibi_df[index_cols].drop_duplicates().shape[0]
        assert base_feat_df.shape[0] == n_unique_segments


class TestAudaceDataLoader:

    @staticmethod
    def test_get_ibi_df(example_data_paths):
        train_loader = AudaceDataLoader(root=example_data_paths["data_dir"])
        in_data = train_loader.get_ibi_df()

        assert in_data is not None

    @staticmethod
    def test_get_split_pred_df(example_data_paths):
        train_loader = AudaceDataLoader(root=example_data_paths["data_dir"])

        in_data = train_loader.get_ibi_df()

        # Use original feature values
        # rather than those relative to each participant's baseline
        rel_values = False
        # Do not use spreadsheet with already extracted features
        load_from_file = False
        # Save spreadsheet with features
        save_file = True

        # Load "out_train" dictionary containing:
        # X: all possible input features
        # y: ground truth stress/rest labels
        # method: method used for heartbeat extraction
        # signal: type of heartbeat signal (ECG, IEML, etc.)
        # sub: participant labels
        # task: task labels (rest periods are labeled with the task that came after)
        # subseg_data: if there are multiple segments corresponding to the same
        # set of labels (e.g. with smaller segment lengths or error augmentation),
        # their indices
        out_train = train_loader.get_split_pred_df(
            load_from_file=load_from_file,
            save_file=save_file,
            rel_values=rel_values,
            in_data=in_data,
        )

        assert out_train is not None
