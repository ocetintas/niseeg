import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import scipy.io

class EmotionEEG(Dataset):
  def __init__(self, datapath):
    self.files = glob.glob(datapath + '*')
    self.load()
  def load(self):
    self.EEG=torch.empty(1280,32,8064)
    self.label = torch.empty(1280,4)
    trial_count = 0
    for f in self.files:
      data=scipy.io.loadmat(f)
      eeg=torch.from_numpy(data['data'][:,:32,:])
      label=torch.from_numpy(data['labels'])
      self.EEG[trial_count:trial_count+eeg.shape[0]]=eeg
      self.label[trial_count:trial_count+eeg.shape[0]]=label
      trial_count+=eeg.shape[0]
      if trial_count >=100:
        break

  def __len__(self):
    return self.data.shape[0]
  def __getitem__(self, index):
    return {'eeg':self.EEG[index], 'label':self.label[index]}
