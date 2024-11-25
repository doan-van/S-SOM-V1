# **SSOM User Guide**

The `SSOM` class implements a **Self-Organizing Map (SOM)**, a type of artificial neural network used for unsupervised learning. This manual explains each method and argument in the class.

---
## **Initialization of SSOM class**

```python
som = SSOM(grid_size=10, num_iterations=100, max_learning_rate=0.1, 
     learning_decay=0.25, neighborhood_decay=0.5, metric="euclidean", 
     grid_shape="2D", log_interval=None)
```

### **Arguments:**

- **`grid_size`**  
  - Specifies the size of the SOM grid.  
  - For `grid_shape="1D"`, it is an integer (number of nodes in the 1D grid).  
  - For `grid_shape="2D"`, it is a tuple `(rows, cols)` specifying the grid dimensions.

- **`num_iterations`**  
  - Total number of iterations during training.  
  - Determines how many steps the SOM will take to adjust to the data.

- **`max_learning_rate`**  
  - The initial learning rate for updating the weights of the SOM.  
  - The learning rate decreases over iterations based on the `learning_decay`.

- **`learning_decay`**  
  - Controls how quickly the learning rate decreases over time.  
  - Higher values mean faster decay.

- **`neighborhood_decay`**  
  - Controls how quickly the neighborhood radius decreases over time.  
  - The radius affects which nodes in the SOM are influenced by a given data point.

- **`metric`**  
  - Specifies the distance metric used to find the **Best Matching Unit (BMU)**:  
    - `"euclidean"`: Uses Euclidean distance.  
    - `"ssim"`: Uses Structural Similarity Index (SSIM) for image data.

- **`grid_shape`**  
  - `"1D"`: Creates a 1D SOM grid (linear topology).  
  - `"2D"`: Creates a 2D SOM grid (grid topology).

- **`log_interval`**  
  - Controls how frequently the training progress is logged and recorded.  
  - If not specified, it is automatically set to ensure no more than 100 logs are saved.


### **Example**

Check **test/** for more examples how to run SSOM

- **`Assume we have sample data (which are two-dimensional data)`**

```python
# Generate synthetic data
import numpy as np
x = np.random.normal((-0.5, -0.5), 0.2, (200, 2))
x = np.append(x, np.random.normal((-0.25, 1), 0.2, (100, 2)), axis=0)
x = np.append(x, np.random.normal((0.5, 0.5), 0.2, (150, 2)), axis=0)
input_data = x
```
- **`Initialize SSOM class (2-D map topology)`**

```python
from ssom import SSOM  # Assuming SSOM is imported correctly
# Initialize a 2D SOM with 10x10 grid
som = SSOM(grid_size=(10, 10), num_iterations=1000, max_learning_rate=0.5, metric="euclidean")
```

- **`Initialize SSOM class (1-D ring topology)`**

```python
# Initialize the SSOM with a 1D grid of size 20
som = SSOM(grid_size=20, num_iterations=5000, grid_shape="1D")
```

### **`Train SSOM with input data`**
```python
som.train(input_data)
```
### **`train()`**
- Trains the SOM on the input data (`input_data`).  
- Training process is to adjusts the weights of the SOM nodes to match the input data distribution.

- **`input_data`**  
  A NumPy array of shape `(samples, dimensions)`. Each row is a data sample.

---

### **Important outputs of `SSOM`**
The `som_grid` and `sample_bmu_mapping` are critical attributes for understanding the result of the SOM training:

1. **`som_grid`**: Represents the learned weights of the SOM nodes and can be visualized to interpret the clustering.
2. **`sample_bmu_mapping`**: Links each input data point to its nearest SOM node, showing how the SOM clusters the data.
   

#### **`som_grid`**

##### **Description**
- The `som_grid` attribute represents the weights of the SOM nodes. It stores the learned representation of the input data.
- Its shape depends on the topology of the SOM:
  - For `grid_shape="1D"`, it is a 2D array of shape `(grid_size, num_input_dims)`.
  - For `grid_shape="2D"`, it is a 3D array of shape `(grid_rows, grid_cols, num_input_dims)`.

##### **Usage**
- After training, the `som_grid` contains the trained SOM nodes, which approximate the distribution of the input data.

##### **Example**
```python
# Access the trained SOM grid
print("Trained SOM Grid:")
print(som.som_grid)
```

---

#### `sample_bmu_mapping`

##### **Description**
- The `sample_bmu_mapping` is a list of tuples, each containing:
  1. The index of a sample in the input data.
  2. The index of the **Best Matching Unit (BMU)** for that sample.
- This mapping shows how each input sample is clustered within the SOM grid.

##### **Usage**
- Use this attribute to understand the clustering of the input data points within the SOM.

##### **Example**
```python
# Access the sample-to-BMU mapping
print("Sample to BMU Mapping:")
for sample_index, bmu_index in som.sample_bmu_mapping:
    print(f"Sample {sample_index} maps to BMU at index {bmu_index}")
```

---


## **Other Methods**


### **1. `_initialize_grid_indices()`**

```python
grid_indices = self._initialize_grid_indices()
```

### **Description**
- Initializes the indices of the SOM grid.  
- For `grid_shape="2D"`, creates a `(rows, cols, 2)` array where each element stores its row and column index.  

### **Example**
```python
# Create 2D grid indices for a 5x5 SOM
som = SSOM(grid_size=(5, 5), grid_shape="2D")
grid_indices = som._initialize_grid_indices()
```

---

### **2. `_initialize_grid()`**

```python
self._initialize_grid(num_input_dims)
```

### **Description**
- Randomly initializes the weights of the SOM grid.  
- For `grid_shape="1D"`, creates a `(grid_size, num_input_dims)` array.  
- For `grid_shape="2D"`, creates a `(rows, cols, num_input_dims)` array.  

### **Arguments**
- **`num_input_dims`**  
  - The number of dimensions in the input data (e.g., 3 for RGB).

### **Example**
```python
# Initialize a 1D SOM grid with 10 nodes and 3 input dimensions
som = SSOM(grid_size=10, grid_shape="1D")
som._initialize_grid(3)
```

---

### **3. `ssim()`**

```python
self.ssim(x, y, C1=0.01**2, C2=0.03**2)
```

### **Description**
- Computes the **Structural Similarity Index (SSIM)** between two data points.  
- Useful for comparing image-like data.

### **Arguments**
- **`x, y`**  
  Input arrays to compare.  
- **`C1, C2`**  
  Small constants to stabilize SSIM computation.

### **Example**
```python
# Compare two RGB patches
ssim_value = som.ssim(patch1, patch2)
```

---

### **4. `find_bmu()`**

```python
bmu_index = self.find_bmu(input_patch)
```

### **Description**
- Finds the **Best Matching Unit (BMU)** for a given input patch based on the specified metric (`"euclidean"` or `"ssim"`).

### **Arguments**
- **`input_patch`**  
  A data point from the input dataset.

### **Returns**
- The index of the BMU:  
  - For `grid_shape="1D"`, an integer index.  
  - For `grid_shape="2D"`, a tuple `(row, col)`.

### **Example**
```python
# Find BMU for an input point
bmu = som.find_bmu(np.array([0.5, 0.5]))
```

---



### **Example**
```python
# Train the SOM with 2D input data
data = np.random.rand(100, 2)
som.train(data)
```

---

## **Attributes for Logging**

The following attributes store historical data during training:

- **`som_grid_history`**  
  A list of SOM grid states logged at each `log_interval`.

- **`neighborhood_influence_history`**  
  A list of neighborhood influence values for each iteration.

- **`learning_rate_history`**  
  A list of learning rates at each `log_interval`.

- **`bmu_history`**  
  A list of BMU indices for selected data points at each `log_interval`.

- **`step_history`**  
  A list of iteration steps corresponding to the logs.

- **`selected_data_point_history`**  
  A list of input data points selected at each `log_interval`.

### **Example**
```python
# Access learning rate history after training
print(som.learning_rate_history)
```

---

## **Visualization Example**

You can visualize the SOM training progress and final results using `matplotlib`.

```python
# Plot learning rate decay
plt.plot(som.step_history, som.learning_rate_history)
plt.title("Learning Rate Decay")
plt.xlabel("Iteration Step")
plt.ylabel("Learning Rate")
plt.grid()
plt.show()
```
