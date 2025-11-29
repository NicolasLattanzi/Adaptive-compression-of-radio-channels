import torch
from torch.utils.data import *

from scipy.signal import ShortTimeFFT
import os
import h5py

num_samples_to_load = 200000


class RadioDataset(Dataset):

    def __init__(self):
        self.path = '../radio_archive'
        self.main_data_file = 'GOLD_XYZ_OSC.0001_1024.hdf5'
        self.radio_channels = []
        self.h5file_path = os.path.join(self.path, self.main_data_file)

        with h5py.File(self.h5file_path, 'r') as f:
            self.X = f['X'][:num_samples_to_load]
            self.Y = f['Y'][:num_samples_to_load]
            self.Z = f['Z'][:num_samples_to_load]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.X[idx] # 1024 x 2 array
        return sample


def train_test_split(dataset, train=0.66, test=0.34):
    return torch.utils.data.random_split(dataset, [train, test])


def apply_STFT():
    ...
    ShortTimeFFT(win=225, hop=128, fft_mode='onesided', mfft=225)


data = RadioDataset()