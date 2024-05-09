from torch.utils.data import Dataset
import pandas as pd


class PathsDataset(Dataset):
    """LibriSpeech paths dataset."""

    def __init__(self, tsv_file, transform=None):
        """
        Args:
            tsv_file (string): Path to the tsv file.
            root_dir (string): Directory with all the audio samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(tsv_file, 'r') as f:
            self.base_path = f.readline().strip()

        self.paths_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, header=None, names=['relative_path', 'duration'])
        self.transform = transform

    def __len__(self):
        return len(self.paths_df)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            relative_path = self.paths_df.iloc[idx]['relative_path']
            return f"{self.base_path}\\{relative_path}"
        else:
            raise ValueError("Invalid index type.")
