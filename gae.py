import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
import numpy as np

class GAE(nn.Module):
    def __init__(self, in_channels, out_channels=2, hidden_dim=64, dropout=0.5):
        super(GAE, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_channels)
        self.bn2 = BatchNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(out_channels, in_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x, edge_index))
        print("Size of input data:", x.size())
        print("Size of embeddings:", x.size())
        print("Embeddings:", x)
        return x

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decoder(z)
    
    
def train_gae_with_early_stopping(node_features, adj_matrix, epochs=700, lr=0.04, weight_decay=5e-4, patience=20):
    in_channels = node_features.shape[1]
    
    # Define the model inside the training function
    model = GAE(in_channels=in_channels, hidden_dim=64, dropout=0.4).to(device)
    
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long).to(device)
    node_features = torch.tensor(node_features, dtype=torch.float).to(device)
    data = Data(x=node_features, edge_index=edge_index).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        reconstructed = model.decoder(z)
        
        loss = criterion(reconstructed, data.x)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    return model, data

def calculate_mse(original, reconstructed):
    mse = nn.MSELoss()
    return mse(reconstructed, original).item()

# Set the device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage
node_features = np.random.rand(10, 3)
adj_matrix = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
])

# Train the model
model, data = train_gae_with_early_stopping(node_features, adj_matrix, epochs=700, lr=0.04, weight_decay=1e-4, patience=20)

# Perform a forward pass and calculate RMSE
model.eval()
with torch.no_grad():
    embeddings = model.encode(data.x, data.edge_index)
    reconstructed = model.decoder(embeddings)
    
    print("Original Node Features:")
    print(data.x)
    print("\nReconstructed Node Features:")
    print(reconstructed)
    print("Size of input data:", data.x.size())
    print("Size of embeddings:", embeddings.size())
    print("Embeddings:", embeddings)

    rmse_value = torch.sqrt(torch.tensor(calculate_mse(data.x.cpu(), reconstructed.cpu()))).item()
    print("RMSE = ", rmse_value)
