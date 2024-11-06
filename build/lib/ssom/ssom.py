import numpy as np
import random as rand

class SSOM:
    def __init__(self, grid_size=10, num_iterations=100, 
                 max_learning_rate=0.1, learning_decay=0.25, neighborhood_decay=0.5, 
                 metric="euclidean", grid_shape="2D", log_interval=None):
        """
        Initialize the Self-Organizing Map (SOM) with specified parameters.
        
        Parameters:
        - grid_size: int or tuple, grid size for the SOM. If `grid_shape` is "1D", grid_size is int.
          If `grid_shape` is "2D", grid_size is a tuple (grid_rows, grid_cols).
        - num_iterations: int, number of training iterations.
        - max_learning_rate: float, initial learning rate.
        - learning_decay: float, decay rate for learning rate.
        - neighborhood_decay: float, decay rate for neighborhood function.
        - metric: str, "euclidean" or "ssim" to specify the similarity metric.
        - grid_shape: str, "1D" for a 1-dimensional grid or "2D" for a 2-dimensional grid.
        - log_interval: int, interval for logging the outputs during training.
        """
        self.grid_shape = grid_shape
        self.num_iterations = num_iterations
        self.max_learning_rate = max_learning_rate
        self.learning_decay = learning_decay
        self.neighborhood_decay = neighborhood_decay
        self.metric = metric

        # Set default log_interval to keep total logs under 100, unless specified by user
        max_logs = 100
        if log_interval is None or log_interval > num_iterations:
            self.log_interval = max(1, num_iterations // max_logs)
        else:
            self.log_interval = log_interval
            
                
        # Initialize logging lists to store historical values
        self.som_grid_history = []
        self.neighborhood_influence_history = []
        self.learning_rate_history = []
        self.bmu_history = []  # To store BMU history at each logging interval
        self.step_history = []  # To store the iterative step number at each logging interval
        self.selected_data_point_history = []  # History of selected data points
        self.sample_bmu_mapping = []  # Initialize the attribute for storing BMU mappings

        if grid_shape == "1D":
            self.grid_size = grid_size  
            self.som_grid = None  
            self.grid_indices = np.arange(self.grid_size)
        elif grid_shape == "2D":
            self.grid_rows, self.grid_cols = grid_size  
            self.som_grid = None  
            self.grid_indices = self._initialize_grid_indices()
        else:
            raise ValueError("grid_shape must be either '1D' or '2D'")
        
        # Print initial settings
        self._print_initial_settings()

    def _print_initial_settings(self):
        """Prints the initial settings of the SSOM instance."""
        print("SSOM Initial Settings:")
        print(f"  Grid Shape: {self.grid_shape}")
        if self.grid_shape == "2D":
            print(f"  Grid Size: {self.grid_rows}x{self.grid_cols}")
        else:
            print(f"  Grid Size: {self.grid_size}")
        print(f"  Number of Iterations: {self.num_iterations}")
        print(f"  Max Learning Rate: {self.max_learning_rate}")
        print(f"  Learning Decay: {self.learning_decay}")
        print(f"  Neighborhood Decay: {self.neighborhood_decay}")
        print(f"  Metric: {self.metric}")
        print(f"  Log Interval: {self.log_interval}")
        print()

    def _initialize_grid_indices(self):
        """Create an index map for the 2D grid, storing row and column indices for each cell."""
        grid_indices = np.zeros([self.grid_rows, self.grid_cols, 2])
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                grid_indices[row, col] = [row, col]
        return grid_indices

    def _initialize_grid(self, num_input_dims):
        """Initialize the SOM grid with random weights."""
        if self.grid_shape == "1D":
            self.som_grid = np.random.uniform(low=0, high=1, size=(self.grid_size, num_input_dims))
        elif self.grid_shape == "2D":
            self.som_grid = np.random.uniform(low=0, high=1, size=(self.grid_rows, self.grid_cols, num_input_dims))
        print("Initialized SOM grid with random weights.")

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
        - bmu_index: int for 1D grid, or tuple for 2D grid (row, col).
        """
        bmu_index = None
        highest_ssim = -1
        min_distance = np.inf

        if self.grid_shape == "1D":
            for i in range(self.grid_size):
                current_patch = self.som_grid[i]
                
                if self.metric == "ssim":
                    current_ssim = self.ssim(input_patch, current_patch)
                    if current_ssim > highest_ssim:
                        highest_ssim = current_ssim
                        bmu_index = i
                        
                elif self.metric == "euclidean":
                    current_distance = np.linalg.norm(current_patch - input_patch)
                    if current_distance < min_distance:
                        min_distance = current_distance
                        bmu_index = i

        elif self.grid_shape == "2D":
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
        
        Sets:
        - self.sample_bmu_mapping: list of tuples, each containing (sample_index, BMU coordinates).
        """
        num_samples = input_data.shape[0]
        num_input_dims = input_data.shape[1]
        self._initialize_grid(num_input_dims)
        
        learning_rate_decay_time = self.num_iterations * self.learning_decay
        initial_neighborhood_radius = (self.grid_size if self.grid_shape == "1D" else self.grid_rows) * self.neighborhood_decay

        for iteration in range(self.num_iterations):
            random_index = rand.randint(0, num_samples - 1)
            selected_patch = input_data[random_index]
            
            bmu_index = self.find_bmu(selected_patch)
            
            current_learning_rate = self.max_learning_rate * np.exp(-iteration / learning_rate_decay_time)
            current_neighborhood_radius = initial_neighborhood_radius * np.exp(-iteration / learning_rate_decay_time)
            
            if self.grid_shape == "1D":
                distance_from_bmu = np.abs(self.grid_indices - bmu_index)
                neighborhood_influence = np.exp(-distance_from_bmu**2 / (2 * current_neighborhood_radius**2))
                self.som_grid += current_learning_rate * neighborhood_influence[:, np.newaxis] * (selected_patch - self.som_grid)

            elif self.grid_shape == "2D":
                distance_from_bmu = np.linalg.norm(self.grid_indices - bmu_index, axis=2)
                neighborhood_influence = np.exp(-distance_from_bmu**2 / (2 * current_neighborhood_radius**2))
                self.som_grid += current_learning_rate * neighborhood_influence[:, :, np.newaxis] * (selected_patch - self.som_grid)

            # Log the step, BMU, som_grid, neighborhood_influence, learning_rate, and selected data point at each interval
            if iteration % self.log_interval == 0 or iteration == self.num_iterations - 1:

                # Calculate the progress bar length based on the current iteration
                progress_length = 50  # Length of the progress bar
                filled_length = int(round(progress_length * (iteration + 1) / self.num_iterations))  # Round to avoid leftover "-"
                bar = '=' * filled_length + '-' * (progress_length - filled_length)
                
                # Print progress bar, current iteration, and learning rate in one line
                print(f"\rIter {iteration + 1}/{self.num_iterations} [{bar}]  LearnRate: {current_learning_rate:.4f}", end='', flush=True)


                self.step_history.append(iteration)
                self.bmu_history.append(bmu_index)
                self.som_grid_history.append(np.copy(self.som_grid))
                self.neighborhood_influence_history.append(neighborhood_influence)
                self.learning_rate_history.append(current_learning_rate)
                self.selected_data_point_history.append(selected_patch)  # Log selected data point

            # Print a new line when the training is complete
            if iteration == self.num_iterations - 1:
                print()  # Move to the next line after the final iteration
    
        # Map each input sample to its BMU and store it in self.sample_bmu_mapping
        self.sample_bmu_mapping = []
        for i, sample in enumerate(input_data):
            bmu_index = self.find_bmu(sample)
            self.sample_bmu_mapping.append((i, bmu_index))


    




