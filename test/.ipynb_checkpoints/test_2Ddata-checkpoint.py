from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# Generate synthetic data
x = np.random.normal((-0.5, -0.5), 0.2, (200, 2))
x = np.append(x, np.random.normal((-0.25, 1), 0.2, (100, 2)), axis=0)
x = np.append(x, np.random.normal((0.5, 0.5), 0.2, (150, 2)), axis=0)

from ssom import SSOM  # Assuming SSOM is imported correctly
# Initialize the SSOM with a 10x10 2D grid
som = SSOM(grid_size=(10, 10), num_iterations=5000, grid_shape="2D")
som.train(x)



# Directory to save temporary images
if not os.path.exists("temp_plots"):
    os.makedirs("temp_plots")

# List to store file paths of saved images for GIF creation
filenames = []

# Plot 1: Input Data with clusters
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], label="Data Points", c="gray", alpha=0.6)
plt.title("Input Data with Clusters")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
filename = "temp_plots/plot_input_data.png"
plt.savefig(filename)
filenames.append(filename)
plt.close()

# Plot 2: Historical SOM Grid with selected data point connection to BMU
for i, (som_grid, selected_patch, bmu_index) in enumerate(zip(som.som_grid_history, som.selected_data_point_history, som.bmu_history)):
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], label="Data Points", c="gray", alpha=0.6)
    
    # Flatten the SOM grid to 2D points for plotting
    som_grid_points = som_grid.reshape(-1, 2)
    plt.scatter(som_grid_points[:, 0], som_grid_points[:, 1], c="blue", marker="s", label="SOM Nodes")
    
    # Draw lines between neighboring nodes in the 2D grid
    for row in range(som.grid_rows):
        for col in range(som.grid_cols):
            if col + 1 < som.grid_cols:  # Horizontal neighbor
                plt.plot([som_grid[row, col, 0], som_grid[row, col + 1, 0]],
                         [som_grid[row, col, 1], som_grid[row, col + 1, 1]], "k-", linewidth=0.5)
            if row + 1 < som.grid_rows:  # Vertical neighbor
                plt.plot([som_grid[row, col, 0], som_grid[row + 1, col, 0]],
                         [som_grid[row, col, 1], som_grid[row + 1, col, 1]], "k-", linewidth=0.5)

    # Highlight connection between the selected data point and its BMU
    bmu_row, bmu_col = bmu_index
    bmu_pos = som_grid[bmu_row, bmu_col]
    plt.plot([selected_patch[0], bmu_pos[0]], [selected_patch[1], bmu_pos[1]], "r-", alpha=0.7, label="Selected Patch to BMU")
    plt.scatter(selected_patch[0], selected_patch[1], c="red", marker="x", label="Selected Data Point", s=100)

    plt.title(f"SOM Grid - Iteration Step {som.step_history[i]}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)

    filename = f"temp_plots/plot_historical_{i}.png"
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Plot 3: Final Results - SOM Grid fitted to data
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], label="Data Points", c="gray", alpha=0.6)
# Flatten the final SOM grid to 2D points for plotting
final_som_grid_points = som.som_grid.reshape(-1, 2)
plt.scatter(final_som_grid_points[:, 0], final_som_grid_points[:, 1], c="blue", marker="s", label="SOM Nodes")

# Draw lines between neighboring nodes in the 2D grid in the final plot
for i in range(som.grid_rows):
    for j in range(som.grid_cols):
        if j + 1 < som.grid_cols:  # Horizontal neighbor
            plt.plot([som.som_grid[i, j, 0], som.som_grid[i, j + 1, 0]],
                     [som.som_grid[i, j, 1], som.som_grid[i, j + 1, 1]], "k-", linewidth=0.5)
        if i + 1 < som.grid_rows:  # Vertical neighbor
            plt.plot([som.som_grid[i, j, 0], som.som_grid[i + 1, j, 0]],
                     [som.som_grid[i, j, 1], som.som_grid[i + 1, j, 1]], "k-", linewidth=0.5)

plt.title("Final SOM Grid fitted to Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)

filename = "temp_plots/plot_final.png"
plt.savefig(filename)
filenames.append(filename)
plt.close()


# Plot the learning rate history
plt.figure(figsize=(8, 6))
plt.plot(som.step_history, som.learning_rate_history, marker="o", linestyle="-", color="b", label="Learning Rate")
plt.title("Learning Rate Decay Over Training Iterations")
plt.xlabel("Iteration Step")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
filename = "temp_plots/plot_learning_rate.png"
plt.savefig(filename)
filenames.append(filename)
plt.close()

# Plot the neighborhood influence history
plt.figure(figsize=(8, 6))
for i, neighborhood_influence in enumerate(som.neighborhood_influence_history):
    avg_influence = np.mean(neighborhood_influence)
    plt.plot(som.step_history[i], avg_influence, "ro")
plt.title("Neighborhood Influence Decay Over Training Iterations")
plt.xlabel("Iteration Step")
plt.ylabel("Average Neighborhood Influence")
plt.grid(True)
filename = "temp_plots/plot_neighborhood_influence.png"
plt.savefig(filename)
filenames.append(filename)
plt.close()

# Create the GIF from the saved images
images = [Image.open(f) for f in filenames]
gif_path = "som_training.gif"
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)

# Cleanup temporary files
for filename in filenames:
    os.remove(filename)
os.rmdir("temp_plots")

print(f"GIF created at: {gif_path}")
