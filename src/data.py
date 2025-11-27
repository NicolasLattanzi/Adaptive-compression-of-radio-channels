import torch
from torch.utils.data import *

from scipy.signal import ShortTimeFFT


class RadioDataset(Dataset):

    def __init__(self):
        self.path = '../radio_archive'

        self.radio_channels = ...

    def __len__(self):
        return len(self.radio_channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return ...


def train_test_split(dataset, train=0.66, test=0.34):
    return torch.utils.data.random_split(dataset, [train, test])


def apply_STFT():
    ...
    ShortTimeFFT(win=225, hop=128, fft_mode='onesided', mfft=225)