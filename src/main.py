from torch.utils.data import DataLoader

import data
from model import GNN

###### hyper parameters ########

batch_size = 16
num_epochs = 8
learning_rate = 0.001

###############################

dataset = data.RadioDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = GNN()
model = torch_geometric.compile(model)

###### training loop

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print('start training')

for epoch in num_epochs:
    train_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # assume batch.y present
        loss = torch.nn.functional.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()

print('end training')