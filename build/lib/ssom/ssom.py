import numpy as np
import random as rand

class SSOM:
    def __init__(self, grid_rows=10, grid_cols=10, num_iterations=100, 
                 max_learning_rate=0.1, learning_decay=0.25, neighborhood_decay=0.5, metric="euclidean"):
        """
        Initialize the Self-Organizing Map (SOM) with specified parameters.
        
        Parameters:
        - grid_rows: int, number of rows in the SOM grid.
        - grid_cols: int, number of columns in the SOM grid.
        - num_iterations: int, number of training iterations.
        - max_learning_rate: float, initial learning rate.
        - learning_decay: float, decay rate for learning rate.
        - neighborhood_decay: float, decay rate for neighborhood function.
        - metric: str, "euclidean" or "ssim" to specify the similarity metric.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_iterations = num_iterations
        self.max_learning_rate = max_learning_rate
        self.learning_decay = learning_decay
        self.neighborhood_decay = neighborhood_decay
        self.metric = metric
        self.som_grid = None
        self.grid_indices = self._initialize_grid_indices()

    def _initialize_grid_indices(self):
        """Create an index map for the grid, storing row and column indices for each cell."""
        grid_indices = np.zeros([self.grid_rows, self.grid_cols, 2])
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                grid_indices[row, col] = [row, col]
        return grid_indices

    def _initialize_grid(self, num_input_dims):
        """Initialize the SOM grid with random weights."""
        self.som_grid = np.random.uniform(low=0, high=1, size=(self.grid_rows, self.grid_cols, num_input_dims))

    def ssim(self, x, y, C1=0.01**2, C2=0.03**2):
        """Compute the Structural Similarity Index (SSIM) between two 2D numpy arrays."""
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        ssim_index = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_index

    def find_bmu(self, input_patch):
        """
        Find the Best Matching Unit (BMU) in the SOM grid based on the specified metric.
        
        Parameters:
        - input_patch: numpy array.
        
        Returns:
        - bmu_index: tuple, the row and column index of the BMU.
        """
        bmu_index = None
        highest_ssim = -1  # For SSIM, initialize to minimum value
        min_distance = np.inf  # For Euclidean distance, initialize to maximum value

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                current_patch = self.som_grid[row, col]
                
                if self.metric == "ssim":
                    current_ssim = self.ssim(input_patch, current_patch)
                    if current_ssim > highest_ssim:
                        highest_ssim = current_ssim
                        bmu_index = (row, col)
                        
                elif self.metric == "euclidean":
                    current_distance = np.linalg.norm(current_patch - input_patch)
                    if current_distance < min_distance:
                        min_distance = current_distance
                        bmu_index = (row, col)
                        
        return bmu_index

    def train(self, input_data):
        """
        Train the SOM on the input data using the specified metric.
        
        Parameters:
        - input_data: numpy array, input data to be clustered (samples, dims).
        
        Returns:
        - sample_bmu_mapping: list of tuples, each containing (sample_index, BMU coordinates).
        """
        num_samples = input_data.shape[0]
        num_input_dims = input_data.shape[1]
        self._initialize_grid(num_input_dims)
        
        learning_rate_decay_time = self.num_iterations * self.learning_decay
        initial_neighborhood_radius = self.grid_rows * self.neighborhood_decay

        # Training loop
        for iteration in range(self.num_iterations):
            # Step 1: Select a random input patch
            random_index = rand.randint(0, num_samples - 1)
            selected_patch = input_data[random_index]
            
            # Step 2: Find Best Matching Unit (BMU) based on chosen metric
            bmu_index = self.find_bmu(selected_patch)
            
            # Step 3: Update learning rate and neighborhood radius
            current_learning_rate = self.max_learning_rate * np.exp(-iteration / learning_rate_decay_time)
            current_neighborhood_radius = initial_neighborhood_radius * np.exp(-iteration / learning_rate_decay_time)
            
            # Step 4: Calculate neighborhood influence
            distance_from_bmu = np.linalg.norm(self.grid_indices - bmu_index, axis=2)
            neighborhood_influence = np.exp(-distance_from_bmu**2 / (2 * current_neighborhood_radius**2))

            # Step 5: Update SOM grid
            self.som_grid += current_learning_rate * neighborhood_influence[:, :, np.newaxis] * (selected_patch - self.som_grid)


        # After training, assign each input sample to its BMU on the trained grid
        sample_bmu_mapping = []
        for i, sample in enumerate(input_data):
            bmu_index = self.find_bmu(sample)
            sample_bmu_mapping.append((i, bmu_index))

        return sample_bmu_mapping