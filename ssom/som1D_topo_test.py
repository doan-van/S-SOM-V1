import numpy as np
import random as rand
import matplotlib.pyplot as plt

# Parameters
grid_size = 20               # Size of the 1D grid
num_iterations = 500        # Number of training iterations
max_learning_rate = 0.5      # Initial learning rate
learning_decay = 0.25        # Decay rate for learning rate
neighborhood_decay = .5     # Decay rate for neighborhood radius
log_interval = 10           # Interval for logging output

# Generate synthetic 2D data
x = np.random.normal((-0.5, -0.5), 0.2, (100, 2))
x = np.append(x, np.random.normal((-0.25, 1), 0.2, (50, 2)), axis=0)
x = np.append(x, np.random.normal((0.5, 0.5), 0.2, (50, 2)), axis=0)

# Initialize the SOM grid and other necessary variables
num_samples, num_input_dims = x.shape
som_grid = np.random.uniform(low=0, high=1, size=(grid_size, num_input_dims))
grid_indices = np.arange(grid_size)

# Set up the figure for live updates
#plt.ion()
#fig, ax = plt.subplots(figsize=(6, 6))

# Training loop with live plotting
learning_rate_decay_time = num_iterations * learning_decay
initial_neighborhood_radius = grid_size * neighborhood_decay

bmu = list(range(grid_size))
for iteration in range(num_iterations):
    # Select a random input sample
    random_index = rand.randint(0, num_samples-1)
    selected_patch = x[random_index]
    
    # Calculate decayed learning rate and neighborhood radius
    current_learning_rate = max_learning_rate * np.exp(-iteration / learning_rate_decay_time)
    current_neighborhood_radius = initial_neighborhood_radius  * np.exp(-iteration / learning_rate_decay_time)
 
    unique_values, frequencies = np.unique(bmu, return_counts=True)
    freq = frequencies / frequencies.sum()
    # Find the index of the minimum frequency using argmin
    #min_frequency_index = np.argmin(frequencies)
    # Sort frequencies in ascending order and get the sorted indices
    sorted_indices = np.argsort(frequencies)
    print(iteration, sorted_indices, np.sort(freq) )
    current_learning_rate_x = current_learning_rate
    for i in range(len(som_grid)-1)[:-1]:
        current_learning_rate_x = current_learning_rate_x*.5
        som_grid[sorted_indices[i]] = som_grid[sorted_indices[i]] + (som_grid[sorted_indices[i+1]]-som_grid[sorted_indices[i]]) * current_learning_rate_x 
    
    #    som_grid[sorted_indices[i]] = som_grid[sorted_indices[i]]  + (selected_patch - som_grid[sorted_indices[i]]) * current_learning_rate
    #som_grid[sorted_indices[0]] = som_grid[sorted_indices[0]] + (som_grid[sorted_indices[1]]-som_grid[sorted_indices[0]]) * current_learning_rate 
    #som_grid[sorted_indices[1]] = som_grid[sorted_indices[1]] + (som_grid[sorted_indices[2]]-som_grid[sorted_indices[1]]) * current_learning_rate *.5
    
    
    # Find the Best Matching Unit (BMU)
    distances = np.linalg.norm(som_grid - selected_patch, axis=1)
    bmu_index = np.argmin(distances)
    bmu.append(bmu_index)
    
   
    if 0:
        # Calculate neighborhood influence
        distances_from_bmu_x = np.linalg.norm(som_grid - som_grid[bmu_index], axis=1)
        # Rank indices from largest to smallest based on distances_from_bmu
        ranked_indices = np.argsort(distances_from_bmu_x)
        grid_indices = ranked_indices
        som_grid = som_grid[ranked_indices]
    
    distance_from_bmu = np.abs(grid_indices - bmu_index)
    #distance_from_bmu = np.abs(ranked_indices - bmu_index)
    neighborhood_influence = np.exp(-distance_from_bmu**2 / (2 * current_neighborhood_radius**2))
    #neighborhood_influence = neighborhood_influence[ranked_indices]
    som_grid0 = som_grid
    # Update weights in the SOM grid
    som_grid = som_grid + current_learning_rate * neighborhood_influence[:, np.newaxis] * (selected_patch - som_grid)
    diff = som_grid - som_grid0
    # Log data and update plots at specified intervals
    if iteration % log_interval == 0 or iteration == num_iterations - 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.clear()
        
        # Plot data points and SOM nodes
        ax.scatter(x[:, 0], x[:, 1], color="gray", alpha=0.6, label="Data Points")
        ax.scatter(som_grid[:, 0], som_grid[:, 1], color="blue", label="SOM Nodes")
        
        # Plot connections between SOM nodes
        for i in range(grid_size - 1):
            ax.plot([som_grid[i, 0], som_grid[i + 1, 0]], [som_grid[i, 1], som_grid[i + 1, 1]], "k-")
        
        # Plot the selected data point in red
        ax.scatter(selected_patch[0], selected_patch[1], color="red", marker="x", s=100, label="Selected Data Point")
        
        # Draw line from the selected data point to the BMU
        bmu_position = som_grid[bmu_index]
        ax.plot([selected_patch[0], bmu_position[0]], [selected_patch[1], bmu_position[1]], "r--", label="Patch to BMU")
        
        # Configure plot details
        ax.set_title(f"SOM Grid at Iteration {iteration}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
        ax.grid(True)
        
        plt.show()
#plt.show()
