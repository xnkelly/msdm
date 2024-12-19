import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time

N = 1024
T = 10**6
queue_lengths = [2, 3, 5, 1024]
np.random.seed(42)  # for reproducibility

def simulate_LL2(N, T, queue_lengths):
    """
    Simulate the Limited Learning (LL2) strategy with restaurant queue lengths and agent behavior.
    Parameters:
        N (int): Number of agents and restaurants
        T (int): Number of simulation days
        queue_lengths (list): List of queue lengths for each simulation scenario
    Returns:
        fractions (array): Fraction of people getting lunch for each day and queue length
    """
    fractions = np.zeros((len(queue_lengths), T))  # Store fractions for each queue length and day
    last_restaurant_queues = np.zeros(N, dtype=int)
    start_time=time.time()
    cur_time=time.time()
    # -------------------------------------------------------------
    # Simulation loop for each queue length
    # -------------------------------------------------------------
    for q_idx, q_star in enumerate(queue_lengths):
        # Each day, run the simulation
        for day in range(T):
            # Initialize agent states and restaurant queues for each day
            restaurant_queues = np.zeros(N, dtype=int)  # Track the number of agents in each restaurant
            # restaurant_queues.fill(0)
            available_restaurants = np.arange(N)  # List of all restaurants initially available
            
            # --- First day: Randomly assign restaurants ---
            if day == 0:
                for agent in range(N):
                    available_restaurants = available_restaurants[restaurant_queues[available_restaurants] < q_star]
                    chosen_restaurant = random.choice(available_restaurants)
                    restaurant_queues[chosen_restaurant] += 1

            # --- For subsequent days: Apply LL2 strategy ---
            else:
                for k, q_len in enumerate(last_restaurant_queues):  # k is the restaurant with queue from the previous day
                    # First agent in the queue at restaurant k
                    # Try to select from restaurants 1 to k-1 (higher ranked ones)
                    if q_len>0:
                        available_high_ranked = available_restaurants[available_restaurants < k]
                        if len(available_high_ranked) > 0:
                            chosen_restaurant = random.choice(available_high_ranked)
                        else:
                            # If no higher ranked restaurant is available, choose from any available restaurant
                            chosen_restaurant = random.choice(available_restaurants)
                        restaurant_queues[chosen_restaurant] += 1
                        if restaurant_queues[chosen_restaurant] >= q_star:
                            available_restaurants = available_restaurants[available_restaurants != chosen_restaurant]
    
                        # Other agents in the queue at restaurant k
                        for _ in range(q_len-1):  # Skip the first agent already handled above
                            # Try to select from restaurants k to N (lower ranked ones)
                            available_low_ranked = available_restaurants[available_restaurants >= k]
                            if len(available_low_ranked) > 0:
                                chosen_restaurant = random.choice(available_low_ranked)
                            else:
                                # If no lower ranked restaurant is available, choose from any available restaurant
                                chosen_restaurant = random.choice(available_restaurants)
                            restaurant_queues[chosen_restaurant] += 1
                            if restaurant_queues[chosen_restaurant] >= q_star:
                                available_restaurants = available_restaurants[available_restaurants != chosen_restaurant]

            # --- Calculate the fraction of agents getting lunch ---
            fraction_f = np.sum(restaurant_queues > 0) / N  # Fraction of agents who get lunch
            fractions[q_idx, day] = fraction_f  # Store the result for the current queue length and day
            # Track restaurants from the previous day with non-zero queues
            last_restaurant_queues = np.copy(restaurant_queues)
            # Optionally, print the fraction for the current day
            if ((day + 1) % 1000 == 0):
                print(f"Day {day + 1}, Queue Length: {fraction_f}, time: {time.time() - cur_time}")
                cur_time = time.time()
    print(f"Simulate time: {time.time() - start_time}")        
    return fractions

fractions = simulate_LL2(N, T, queue_lengths)

plt.figure(figsize=(10, 6))

# Define markers and colors for different queue lengths
markers = ['+', 'x', '*', 's', 'o']  # Marker types
color = 'black'  # Uniform coloro

# For each queue length, plot the fraction of occupied restaurants
for i in range(fractions.shape[0]):
    mean_value = np.mean(fractions[i])  # 计算每个 q_idx 的均值
    print(f"mean value of q*={queue_lengths[i]}: {mean_value}")
    # Get unique values and their counts for each day
    unique_values, counts = np.unique(fractions[i], return_counts=True)
    # print(unique_values, counts)
    # Calculate probabilities (frequencies)
    probabilities = counts / T

    # Plot each queue length with different markers
    if markers[i] == 's' or markers[i] == 'o':  # Check if the marker is square
        plt.scatter(unique_values, probabilities, label=f'q* = {queue_lengths[i]}', 
                    marker=markers[i], edgecolor=color, facecolors='none')
    else:
        plt.scatter(unique_values, probabilities, label=f'q* = {queue_lengths[i]}', 
                    marker=markers[i], color=color)
# Add title and labels
plt.title('Distribution of the Fraction of Occupied Restaurants (Modified KPR LL2 Strategy)')
plt.xlabel('Fraction of Occupied Restaurants, f')
plt.ylabel('Probability, D(f)')
plt.legend()
plt.show()
