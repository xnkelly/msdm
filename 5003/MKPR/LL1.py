import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time

N = 1024
T = 10**6
queue_lengths = [2, 3, 5, 1024]
np.random.seed(42)  # for reproducibility

def simulate_LL1(N, T, queue_lengths):
    """
    Simulate the Limited Learning (LL1) strategy with restaurant queue lengths.
    Parameters:
        N (int): Number of agents (equal to number of restaurants)
        T (int): Number of days for simulation
        queue_lengths (list of int): List of queue lengths to simulate
    Returns:
        fraction_f (array): Fraction of people getting lunch for each queue length and day
    """
    fractions = np.zeros((len(queue_lengths), T))  # Store fractions for each queue length and day
    start_time = time.time()
    cur_time = time.time()
    # -------------------------------------------------------------
    # Simulation loop:
    # -------------------------------------------------------------
    for q_idx, q_star in enumerate(queue_lengths):
        # Initialize the agent states and restaurant queues for each day
        agent_best = 0  # Counter for BEST agents
        # Each day, run the simulation
        for day in range(T):
            restaurant_queues = np.zeros(N, dtype=int)  # Track the number of agents in each restaurant
            available_restaurants = list(range(N-1))
            
            # Process best state agents (try to go to the best restaurant)
            for agent in range(N):
                if agent < agent_best and restaurant_queues[-1] < q_star:  # BEST (index: N-1)
                    restaurant_queues[-1] += 1
                else:
                    # If the best restaurant is full, randomly select a restaurant with available space
                    chosen_index = random.randint(0, len(available_restaurants) - 1)
                    chosen_restaurant = available_restaurants[chosen_index]
                    restaurant_queues[chosen_restaurant] += 1
                    if (restaurant_queues[chosen_restaurant] >= q_star):
                        #available_restaurants.remove(available_restaurants[chosen_index])
                        available_restaurants.pop(chosen_index)
                        #idx = available_restaurants.index(chosen_restaurant)
                        #available_restaurants[idx] = available_restaurants[-1]
                        #available_restaurants.pop()
            agent_best = np.sum(restaurant_queues > 0)
            fraction_f = agent_best / N  # Fraction of occupied restaurants
            
            # Store the result for the current queue length and day
            fractions[q_idx, day] = fraction_f
            if ((day+1)%50000==0):
                print(f"Day {day + 1}, Queue Length: {fraction_f}, time: {time.time()-cur_time}")
                cur_time=time.time()
    print(f"Simulate time: {time.time()-start_time}") 
    return fractions

# 模拟不同队列长度下的情况
fractions = simulate_LL1(N, T, queue_lengths)

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
plt.title('Distribution of the Fraction of Occupied Restaurants (Modified KPR LL1 Strategy)')
plt.xlabel('Fraction of Occupied Restaurants, f')
plt.ylabel('Probability, D(f)')
plt.legend()
plt.show()
