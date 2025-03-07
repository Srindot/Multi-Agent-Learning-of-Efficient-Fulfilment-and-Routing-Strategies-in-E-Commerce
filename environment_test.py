# Create an instance of the Environment
env = Environment()

# Test resetting the environment
print("Testing environment reset...")
env.reset()
print(f"Number of warehouses: {len(env.warehouses)}")
print(f"Number of customers: {len(env.customers)}")
print("Warehouses after reset:")
for warehouse in env.warehouses:
    print(f"Warehouse ID: {warehouse.warehouse_id}, Location: {warehouse.location}, Current Inventory: {warehouse.current_inventory}")

print("\nCustomers after reset:")
for customer in env.customers:
    print(f"Customer ID: {customer.customer_id}, Location: {customer.location}, Demand: {customer.demand}, Time Window: {customer.time_window}")

# Test generating customers
print("\nTesting customer generation...")
env.generate_customers()
print(f"Number of customers after generation: {len(env.customers)}")

# Test distance calculation
print("\nTesting distance calculation...")
loc1 = (0, 0)
loc2 = (3, 4)
distance = env.distance(loc1, loc2)
print(f"Distance between {loc1} and {loc2}: {distance}")

# Test creating graph matrices
print("\nTesting graph matrix creation...")
feature_matrix, adjacency_matrix = env.create_graph_matrices()
print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"Adjacency matrix shape: {adjacency_matrix.shape}")

# Test input actions
print("\nTesting input actions...")
c2s_decisions = [(0, 0, 0), (1, 1, 1)]
vrp_decisions = [(0, [0, 1])]
env.input_actions(c2s_decisions, vrp_decisions)
for customer in env.customers:
    print(f"Customer ID: {customer.customer_id}, Fulfilled: {customer.fulfilled}, No. of Deferred: {customer.noof_defered}, Assigned Warehouse: {customer.assigned_warehouse_id}")

# Test reward calculation
print("\nTesting reward calculation...")
c2s_rewards, vrp_rewards = env.calculate_rewards(c2s_decisions, vrp_decisions)
print(f"C2S Rewards: {c2s_rewards}")
print(f"VRP Rewards: {vrp_rewards}")

# Test simulation
print("\nOutput Embeddings...")
print(env.create_graph_matrices())

