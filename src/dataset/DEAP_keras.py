import keras
import numpy as np
import glob
import scipy.io
import os.path as osp


class DEAP:
    def __init__(self, subject=12, num_segments=12, device="cpu", num_classes=3):
        self.subject = subject
        self.num_segments = num_segments
        self.device = device
        self.num_classes = num_classes
        self._load_data()

    def _get_subject_file_names(self, s):
        eeg_root_folder = "/content/drive/My Drive/data_preprocessed_matlab"
        landmarks_root_folder = "/content/drive/My Drive/landmarks"
        subject_code = "s" + "{:02d}".format(s)
        # Get eeg file name
        eeg_file = os.path.join(eeg_root_folder, subject_code + ".mat")
        # Get the landmark file names
        landmark_files = []
        for t in range(1, 41):
            trial_code = "trial" + "{:02d}".format(t)
            trial_file_name = os.path.join(
                landmarks_root_folder,
                subject_code,
                subject_code + "_" + trial_code + ".npy",
            )
            landmark_files.append(trial_file_name)
        return eeg_file, landmark_files

    def _read_eeg_file(self, file, seconds_to_remove=3, sampling_freq=128):
        data = scipy.io.loadmat(file)
        eeg_data = data["data"][
            :, :32, :
        ]  # First 32 channels belong to the EEG (video=40, channel=40, data=8064)
        eeg_data = eeg_data[
            :, :, sampling_freq * seconds_to_remove :
        ]  # Remove baseline recording
        labels = (
            data["labels"] // self.num_classes
        )  # valence, arousal, dominance, liking (video=40, 4)
        assert eeg_data.shape == (
            40,
            32,
            8064 - sampling_freq * seconds_to_remove,
        ), "EEG data reading failed!"
        assert labels.shape == (40, 4), "Labels reading failed!"
        return eeg_data, labels

    def _read_landmark_files(self, files):
        landmark_data = []
        for f in files:
            trial_data = np.load(f)
            trial_data = np.reshape(
                trial_data, (trial_data.shape[0], -1)
            )  # (3000, 81*2)
            trial_data = np.transpose(trial_data)  # (81*2, 3000)
            landmark_data.append(trial_data)
        landmark_data = np.array(
            landmark_data
        )  # (video=40, 81*2, 3000) Compatible with EEG
        assert landmark_data.shape == (
            40,
            81 * 2,
            3000,
        ), "Landmark data reading failed!"
        return landmark_data

    def _epoch_data(self, eeg_data, labels, landmark_data):
        assert (
            eeg_data.shape[0] == landmark_data.shape[0]
        ), "Mismatch between EEG and landmark data"
        assert (
            eeg_data.shape[2] % self.num_segments == 0
            and landmark_data.shape[2] % self.num_segments == 0
        ), "Choose a better number of segments"
        eeg_data_per_segment = int(eeg_data.shape[2] / self.num_segments)
        landmark_data_per_segment = int(landmark_data.shape[2] / self.num_segments)

        eeg, landmark = [], []
        for v in range(eeg_data.shape[0]):
            for i in range(self.num_segments):
                eeg.append(
                    eeg_data[
                        v, :, i * eeg_data_per_segment : (i + 1) * eeg_data_per_segment
                    ]
                )
                landmark.append(
                    landmark_data[
                        v,
                        :,
                        i
                        * landmark_data_per_segment : (i + 1)
                        * landmark_data_per_segment,
                    ]
                )

        self.eeg = np.array(eeg).transpose(0, 2, 1)  # (batch, seq, feature)
        self.landmark = np.array(landmark).transpose(0, 2, 1)  # (batch, seq, feature)
        self.labels = np.repeat(
            labels, self.num_segments, axis=0
        )  # (batch, 4 emotions)

        assert self.eeg.shape == (
            eeg_data.shape[0] * self.num_segments,
            eeg_data_per_segment,
            eeg_data.shape[1],
        ), "EEG epoching went wrong!"
        assert self.landmark.shape == (
            landmark_data.shape[0] * self.num_segments,
            landmark_data_per_segment,
            landmark_data.shape[1],
        ), "Landmark epoching went wrong"
        assert self.labels.shape == (
            labels.shape[0] * self.num_segments,
            4,
        ), "Label epoching went wrong"
        assert (
            self.eeg.shape[0] == self.landmark.shape[0]
            and self.landmark.shape[0] == self.labels.shape[0]
        ), "Number of data does not match!"

    def _load_data(self):
        eeg_file, landmark_files = self._get_subject_file_names(self.subject)
        eeg_data, labels = self._read_eeg_file(eeg_file)
        landmark_data = self._read_landmark_files(landmark_files)
        self._epoch_data(eeg_data, labels, landmark_data)
        self._normalize_landmarks()

    def _normalize_landmarks(self):
        mean = np.mean(self.landmark)
        print(mean)
        std = np.std(self.landmark)
        self.landmark = (self.landmark - mean) / std

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, ix):
        return {
            "eeg": self.eeg[ix],
            "face": self.landmark[ix],
            "label_val": keras.utils.to_categorical(
                self.labels[ix, 0], num_classes=self.num_classes
            ),
            "label_arousal": keras.utils.to_categorical(
                self.labels[ix, 1], num_classes=self.num_classes
            ),
        }

    def train_valid_split(self, split_ratio=0.2):
        split_idx = int(split_ratio * len(self))
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]
        return train_indices, val_indices