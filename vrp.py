import torch
import numpy as np
import torch.nn as nn


class VRPLNetwork(nn.Module):
    def __init__(self, input_dim, output_dim ):
        super(VRPLNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, output_dim)  # Adjusted to output the correct number of actions
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

    def make_decisions(self, env, gae_model, feature_matrix, adjacency_matrix):
        # Convert the input data to tensors
        feature_matrix = torch.tensor(feature_matrix, dtype=torch.float).to(device)
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long).to(device)
        
        # Encode the features to obtain embeddings
        with torch.no_grad():
            embeddings = gae_model.encode(feature_matrix, edge_index)
        state = embeddings.flatten().cpu().numpy()

        # Ensure the state tensor has the correct shape
        state_dim = self.fc1.in_features  # Get the expected input dimension from the VRP model
        state = state[:state_dim]  # Ensure the state has the correct number of features
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)

        # Get the action space
        action_space = get_vrp_action_space(env.customers)
        print(f'Action space size: {len(action_space)}')
        print(f'Q-values output size: {self(state).size()}')

        if len(action_space) == 0:
            print("Error: Action space is empty.")
            return []

        # Get the action for the given embeddings
        with torch.no_grad():
            q_values = self(state)
        action_index = q_values.argmax().item()
        print(f'Action index: {action_index}')

        # Ensure the action index is within bounds
        if action_index >= len(action_space):
            action_index = len(action_space) - 1
        action = action_space[action_index]
        
        return action  # Return the action tuple





