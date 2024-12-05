# Importing Modules
import random as rnd
import numpy as np
from typing import List, Tuple, Dict

class Warehouse:
    def __init__(self, warehouse_id, location, max_inventory):
        self.warehouse_id = warehouse_id
        self.location = location
        self.max_inventory = max_inventory
        self.current_inventory = max_inventory
    
    def restock(self):
        self.current_inventory = self.max_inventory
    
class Customer:
    def __init__(self, customer_id, location, demand, time_window, noof_defered=0):
        self.customer_id = customer_id
        self.location = location
        self.demand = demand
        self.time_window = time_window
        self.fulfilled = False
        self.noof_defered = noof_defered
        self.assigned_warehouse_id = None
        self.assigned_vehicle_id = None  # Initialize with None


class Vehicle:
    def __init__(self, vehicle_id, capacity, speed):
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.speed = speed 
        self.route = [] 
        self.current_load = 0
    
    def reset(self):
        self.route = []
        self.current_load = 0

class Environment:
    def __init__(self, grid_size=200, noof_warehouses=4, noof_customers=400, vehicle_capacity=20, vehicle_speed=2, simulation_time=100, episode_time=5):
        self.grid_size = grid_size
        self.noof_warehouses = noof_warehouses
        self.noof_customers = noof_customers
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_speed = vehicle_speed
        self.simulation_time = simulation_time
        self.episode_time = episode_time
        self.no_of_episodes = simulation_time // episode_time
        self.time_lapsed = 0

        # Setting up warehouses and initializing vehicles and customers
        self.warehouses = self._setting_up_warehouses()
        self.vehicles = []
        self.customers = []

    def Rl_Decision(self, feature_matrix, adjacency_matrix):
        c2s_decisions, vrp_decisions = [], []
        c2s_rewards, vrp_rewards = self.calculate_rewards(c2s_decisions, vrp_decisions)
        return c2s_decisions, vrp_decisions
    
    def simulation_for_each_episode(self):
        self.generate_customers()
        feature_matrix, adjacency_matrix = self.create_graph_matrices()
        self.Rl_Decision(feature_matrix, adjacency_matrix)
        self.time_lapsed += self.episode_time
    
    def simulation(self):
        self.time_lapsed = 0
        self.reset()  # Resetting the environment before starting simulation
        for _ in range(self.no_of_episodes):
            self.simulation_for_each_episode()

    def input_actions(self, c2s_decisions, vrp_decisions):
        # Reset the vehicles list and create instances based on vrp_decisions
        self.vehicles = []
        vehicle_ids = [decision[0] for decision in vrp_decisions]
        noof_vehicles = len(set(vehicle_ids))
        
        # Create vehicles
        for vehicle_id in range(noof_vehicles):
            vehicle = Vehicle(vehicle_id=vehicle_id, capacity=self.vehicle_capacity, speed=self.vehicle_speed)
            self.vehicles.append(vehicle)

        # Process C2S decisions
        for decision in c2s_decisions:
            customer_id, warehouse_id, defer_flag = decision
            customer = self.customers[customer_id]
            warehouse = self.warehouses[warehouse_id]

            if defer_flag:
                customer.noof_defered += 1
                continue

            if warehouse.current_inventory >= customer.demand:
                customer.assigned_warehouse_id = warehouse_id
                warehouse.current_inventory -= customer.demand
                customer.fulfilled = True
            else:
                customer.noof_defered += 1

        # Process VRP decisions for vehicle routing
        for decision in vrp_decisions:
            vehicle_id, route = decision
            vehicle = self.vehicles[vehicle_id]
            vehicle.reset()

            for customer_id in route:
                customer = self.customers[customer_id]
                warehouse_id = customer.assigned_warehouse_id
                warehouse = self.warehouses[warehouse_id]

                if customer.fulfilled and vehicle.current_load + customer.demand <= vehicle.capacity:
                    vehicle.route.append(customer.location)
                    vehicle.current_load += customer.demand
                else:
                    customer.noof_defered += 1
                    customer.fulfilled = False

    def calculate_rewards(self, c2s_decisions, vrp_decisions):
        
        def calculate_c2s_reward(decision):
            a1, a2 = 1, 2
            customer_id, warehouse_id, defer_flag = decision
            customer = self.customers[customer_id]
            warehouse = self.warehouses[warehouse_id]
            
            if defer_flag:
                h = customer.noof_defered
                deferred_reward = a1 * (-2.12) + a2 * (-1)
                return np.power(0.9, h) * deferred_reward
            
            Di = -self.distance(warehouse.location, customer.location)
            trip_customers = sum(1 for c in self.customers if c.assigned_warehouse_id == warehouse_id)
            Li = -Di / max(trip_customers, 1)
            
            Fi = 1 if customer.fulfilled else 0
            Ui = -((self.vehicle_capacity - customer.demand) / self.vehicle_capacity)
            
            return a1 * (Di + Li) + Fi + a2 * Ui

        def calculate_vrp_reward(decision):
            vehicle_id, route = decision
            vehicle = self.vehicles[vehicle_id]
            
            P = len(route)
            route_distance = 0
            vrp_rewards = []

            for p, customer_id in enumerate(route):
                customer = self.customers[customer_id]
                warehouse_id = customer.assigned_warehouse_id
                warehouse = self.warehouses[warehouse_id]
                
                dp = self.distance(vehicle.route[p - 1] if p > 0 else warehouse.location, customer.location)
                tp = dp / vehicle.speed
                
                Rk_p = (0.7 - dp / self.grid_size) + (1.0 - tp / 1.5) + (0.9 * (P - p))
                vrp_rewards.append(Rk_p)
                route_distance += dp
            
            Dreturn = self.distance(vehicle.route[-1], warehouse.location) if route else 0
            Rterm = 2 * 0.7 - 1 / (P + 1) * (route_distance + Dreturn)
            vrp_rewards.append(Rterm)
            
            return vrp_rewards

        c2s_rewards = [calculate_c2s_reward(decision) for decision in c2s_decisions]
        vrp_rewards = [calculate_vrp_reward(decision) for decision in vrp_decisions]
        return c2s_rewards, vrp_rewards

    def _setting_up_warehouses(self):
        warehouse_locations = [
            (-self.grid_size/4, -self.grid_size/4),
            (self.grid_size/4, -self.grid_size/4),
            (-self.grid_size/4, self.grid_size/4),
            (self.grid_size/4, self.grid_size/4)
        ]
        warehouses = []  # Initialize outside loop to store all warehouses
        for i in range(4):
            warehouses.append(Warehouse(warehouse_id=i, location=warehouse_locations[i], max_inventory=100))
        return warehouses
    
    def generate_customers(self):
        customers_per_each_episode = self.noof_customers // self.no_of_episodes
        customers_per_each_episode_rnd = rnd.randint(customers_per_each_episode // 2, customers_per_each_episode) 
        start = len(self.customers)
        if start + customers_per_each_episode_rnd > self.noof_customers:
            raise ValueError(f"Attempting to generate {customers_per_each_episode_rnd} customers exceeds the total allowed {self.noof_customers}.")
        
        for i in range(start, start + customers_per_each_episode_rnd): 
            x = rnd.uniform(-self.grid_size / 2, self.grid_size / 2)
            y = rnd.uniform(-self.grid_size / 2, self.grid_size / 2)
            demand = rnd.randint(1, 10)
            time_window = (rnd.uniform(0.2, 0.8) * self.grid_size, rnd.uniform(0.9, 2.0) * self.grid_size)
            self.customers.append(Customer(customer_id=i, demand=demand, location=(x, y), time_window=time_window))

    def spawn_vehicle(self):
        vehicle = Vehicle(vehicle_id=len(self.vehicles), capacity=self.vehicle_capacity, speed=self.vehicle_speed)
        self.vehicles.append(vehicle)
        return vehicle
    
    def reset(self):
        for warehouse in self.warehouses:
            warehouse.restock()
        self.vehicles = []
        self.customers = []
        self.generate_customers()

    def distance(self, loc1, loc2):
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def norm_1_distance(self, loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def create_graph_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.customers)
        feature_matrix = np.zeros((n, 4))
        adjacency_matrix = np.zeros((n, n))

        for i, customer_i in enumerate(self.customers):
            feature_matrix[i, :] = [
                customer_i.demand, 
                customer_i.location[0], 
                customer_i.location[1], 
                customer_i.time_window[1] - customer_i.time_window[0]
            ]

            for j, customer_j in enumerate(self.customers):
                adjacency_matrix[i, j] = self.norm_1_distance(customer_i.location, customer_j.location)
        
        return feature_matrix, adjacency_matrix