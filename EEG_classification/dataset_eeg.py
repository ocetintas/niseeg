import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import scipy.io


class EmotionEEG(Dataset):
    def __init__(self, datapath, seg_length=640):
        self.files = glob.glob(datapath + "*")
        self.seg_length = seg_length  ##640
        self.load()

    def load(self):
        num_session = len(self.files)
        trial_per_session = 40
        seg_per_session = 8064 // self.seg_length  ## 12
        self.EEG = torch.empty(
            seg_per_session * trial_per_session * num_session, 32, seg_length
        )
        self.label = torch.empty(seg_per_session * trial_per_session * num_session, 4)

        trial_count = 0
        for f in self.files:
            data = scipy.io.loadmat(f)
            eeg = data["data"][:, :32, : seg_length * seg_per_session]
            split_eeg = torch.from_numpy(
                np.array(np.split(eeg, 12, axis=-1)).reshape(-1, 32, seg_length)
            )
            split_label = torch.from_numpy(
                np.tile(data["labels"], (seg_per_session, 1))
            )
            self.EEG[trial_count : trial_count + split_eeg.shape[0]] = split_eeg
            self.label[trial_count : trial_count + split_eeg.shape[0]] = split_label
            trial_count += split_eeg.shape[0]

    def __len__(self):
        return self.EEG.shape[0]

    def __getitem__(self, index):
        return {"eeg": self.EEG[index], "label": self.label[index]}

    def train_valid_split(self, split_ratio=0.2, seed=0):
        np.random.seed(seed)
        split_idx = int(split_ratio * len(self))
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]
        return train_indices, val_indices
