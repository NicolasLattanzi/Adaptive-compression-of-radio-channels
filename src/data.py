import torch


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


def train_test_split(dataset, train=0.6, test=0.4):
    return torch.utils.data.random_split(dataset, [train, test])

