import torch
from torch.utils.data import *

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import os
import h5py
import utils
import numpy as np

'''
The dataset exhibits the following structure:

    - 24 modulations: OOK, ASK4, ASK8, BPSK...
    - 26 SNRs per modulation (-20 dB to +30 dB in steps of 2dB).
    - 4096 frames per modulation-SNR combination.
    - 1024 complex time-series samples per frame.
    - Samples as floating point in-phase and quadrature (I/Q) components, resulting in a (1024,2) frame shape.
    - 2.555.904 frames in total.

Each frame can be retrieved by accessing the HDF5 groups:

    X: I/Q components of the frame;
    Y: Modulation of the frame (one-hot encoded)
    Z: SNR of the frame
'''

num_samples_to_load = 200000


class RadioDataset(Dataset):

    def __init__(self):
        self.path = '../radio_archive'
        self.main_data_file = 'GOLD_XYZ_OSC.0001_1024.hdf5'
        self.h5file_path = os.path.join(self.path, self.main_data_file)

        with h5py.File(self.h5file_path, 'r') as f:
            self.X = f['X'][:num_samples_to_load]
            self.Y = f['Y'][:num_samples_to_load]
            self.Z = f['Z'][:num_samples_to_load]
        
        # (fs=2.048e6 Hz from the RadioML docs)
        self.win = gaussian(225, std=7, sym=True)
        self.STFT = ShortTimeFFT(win=self.win, hop=128, fft_mode='twosided', mfft=225, fs=2.048e6)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.X[idx]   # (1024, 2) array
        signal = sample[:, 0] + 1j * sample[:, 1]
        spectrogram = self.STFT.stft(signal)  # (225, 8) array
        spectrogram_tensor = torch.tensor(np.abs(spectrogram), dtype=torch.float32)
        
        modulation = utils.get_modulation(self.Y[idx])
        snr = self.Z[idx]
        
        # nodes and nodes features
        T_total = 16  # num of desired nodes
        spectrogram_nodes = self._extract_stft_nodes(spectrogram, T_total)
        x = torch.tensor(np.abs(spectrogram_nodes), dtype=torch.float32)  # (16, 225)
        
        # edges and edges features
        edge_index = self._temporal_edges(T_total)
        edge_attr = self._temporal_edge_features(spectrogram_nodes)
        
        # Grafo PyG Data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=modulation, snr=snr)
        graph.num_nodes = T_total
        return graph

    def _extract_stft_nodes(self, spectrogram, T_total):
        # extract T_total time frames from the spectrogram 
        orig_T, F = spectrogram.shape[1], spectrogram.shape[0]  # F=225
        step = max(1, orig_T // T_total)
        
        nodes = []
        for t in range(0, orig_T, step):
            start = max(0, t-1)
            end = min(orig_T, t+2)
            node_spec = np.mean(np.abs(spectrogram[:, start:end]), axis=1)
            nodes.append(node_spec)
        
        # Padding
        nodes = nodes[:T_total] + [nodes[-1]] * (T_total - len(nodes))
        return np.array(nodes)  # (T_total, 225)

    def _temporal_edges(self, num_nodes):
        # Edges: i → i+1 e i+1 → i (bidirectional and ciclic ??)
        src = torch.tensor(range(num_nodes))
        tgt = torch.tensor((range(1, num_nodes) + [0]) % num_nodes)
        return torch.stack([src, tgt], dim=0)

    def _temporal_edge_features(self, spectrogram_nodes):
        diffs = []
        for i in range(len(spectrogram_nodes)):
            j = (i + 1) % len(spectrogram_nodes)
            diff = np.abs(spectrogram_nodes[i] - spectrogram_nodes[j])
            diffs.append(diff)
        return torch.tensor(diffs, dtype=torch.float32)
    
    def reconstruct_signal(self, spectrogram):
        signal_reconstructed = ...
        return signal_reconstructed


def train_test_split(dataset, train=0.66, test=0.34):
    return torch.utils.data.random_split(dataset, [train, test])


from torch_geometric.loader import DataLoader
dataset = RadioDataset()
graph = dataset[0]
print(f"Nodes: {graph.num_nodes}, Features: {graph.x.shape}")  # 16 x 225
print(f"Edge: {graph.edge_index.shape}, y: {graph.y}")  # modulation class
