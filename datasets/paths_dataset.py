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
        self.file = pd.read_csv(tsv_file, sep='\t')
        self.paths = self.file.index
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]
