# import files
from environment import Environment, Customer, Warehouse, Vehicle
from gae import GAE, train_gae_with_early_stopping




import torch
import torch.nn as nn
import numpy as np
import random
import random as rnd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    env = Environment()
    replay_buffer = ReplayBuffer(10000)  # Example replay buffer
    node_features, adj_matrix = env.create_graph_matrices()
    
    gae_model, gae_data = train_gae_with_early_stopping(node_features, adj_matrix)
    dqn_model = DQN(input_dim=19, output_dim=5).to(device)
    
    train_c2s_agent(env, gae_model, dqn_model, replay_buffer)
    
    return gae_model, dqn_model

# Run the combined train function
gae_model, dqn_model = train()

    

# Assuming the following classes and functions are already defined:
# - Environment
# - GAE
# - DQN
# - make_decisions
# - Customer
# - Warehouse
# - Vehicle

def run_simulation(env, gae_model, dqn_model):
    env.time_lapsed = 0
    env.reset()
    episode_rewards = []

    for episode in range(env.no_of_episodes):
        # Generate new customers at the start of each episode
        env.generate_customers()
        feature_matrix, adjacency_matrix = env.create_graph_matrices()

        # Call the make_decisions function
        c2s_decisions = make_decisions(env, gae_model, dqn_model, feature_matrix, adjacency_matrix)
        print(f'Episode {episode + 1}/{env.no_of_episodes}, C2S Decisions: {c2s_decisions}')

        # Calculate the rewards before applying the actions
        try:
            c2s_reward = sum(env.calculate_c2s_reward(decision) for decision in c2s_decisions)
            print(f'C2S Reward: {c2s_reward}')
        except TypeError as e:
            print(f"Error calculating rewards: {e}. Ensure decisions are in (customer_id, warehouse_id, defer_flag) format.")
            continue

        # Apply input actions to update the environment
        env.input_actions(c2s_decisions, [])
        
        # Calculate total reward for the episode
        total_reward = c2s_reward
        episode_rewards.append(total_reward)
        
        env.time_lapsed += env.episode_time
    
    return episode_rewards

# Example usage
env = Environment()
gae_model = GAE(in_channels=7, hidden_dim=64, dropout=0.4).to(device)
dqn_model = DQN(input_dim=19, output_dim=5).to(device)

# Run the simulation and get episode rewards
episode_rewards = run_simulation(env, gae_model, dqn_model)

# Plot the rewards vs number of episodes
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Episode')
plt.show()
