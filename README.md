Radio signals change over time because of noise, fading, interference, and mobility. 
Compression methods should adapt to these changes.
In this project, you convert radio spectrograms into graphs, where each time–frequency bin becomes a node. 
You use a Graph Neural Network (GNN) to learn how much each node can be compressed while keeping the signal quality.

Dataset: DeepSig RadioML 2018.01A, a standard dataset for RF learning.
Architecture (PyTorch Geometric)
