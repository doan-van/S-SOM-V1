# Structural Self Organizing Map Library - Tutorial

Welcome to the **S-SOM** (Structural Self-organizing Map) library! This guide will help you understand how to install and use the library's features effectively, including detailed examples of how to create and manipulate SSOMs using the provided class.

Please cite the paper if you are using this library.

***
Doan, Q.-V., Kusaka, H., Sato, T., and Chen, F.: S-SOM v1.0: a structural self-organizing map algorithm for weather typing, Geosci. Model Dev., 14, 2097–2111, https://doi.org/10.5194/gmd-14-2097-2021, 2021.
***


## Table of Contents

1. [Installation](#installation)
2. [Introduction to SSOM](#introduction-to-ssom)
3. [Getting Started](#getting-started)
4. [Usage Examples](#usage-examples)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)


### Installation

To use the SSOM library, ensure you have Python 3.x installed along with the following dependencies:

```bash
pip install numpy matplotlib
```

You can install the SSOM library from GitHub:

```bash
git clone https://github.com/doan-van/S-SOM-V1.git
```

Navigate to the cloned directory and install the library locally:

```bash
cd S-SOM-V1
pip install .
```

Alternatively, you can download the `ssom.py` file from the GitHub repository and add it to your working directory.

### Introduction to SSOM

**S-SOM** stands for **Structural Self-organizing Map**. SSOM is introduced by Doan et al. (2021) primarily to enhance the ability of SOM in weather pattern detection. 

Similar with SOM, S-SOM is a type of artificial neural network used for clustering and visualizing high-dimensional data. The SSOM algorithm projects input data into a lower-dimensional space, typically in the form of a two-dimensional grid, which reveals the relationships and similarities between data points. The difference between S-SOM and "conventional" SOM is the ability of S-SOM in detect the similarity between "structured" data (i.e., data which having time or space orders) through considering "structural" similarity (instead of simple Euclidean distance) when search the best matching unit.

The main class in this library is `SSOM`, which allows you to:
- Create a new SSOM instance.
- Train the SSOM with data.
- Visualize the results.

### Getting Started

To get started with the SSOM library, import the `SSOM` class and create an instance as shown below:

```python
from ssom import SSOM

# Define the SSOM parameters
som = SSOM(grid_size=(10, 10))
```

- **`grid_size`**: Tuple representing the dimensions of the grid (e.g., 10x10).

### Usage Examples

#### Step 1: Creating and Initializing the SSOM

To begin, create a new instance of the SSOM as described above. The grid size and input dimension should match your use case. For example, if you want to work with RGB images, `input_dim` should be 3.

```python
# Initialize a 2D SSOM with a 10x10 grid 
som = SSOM(grid_size=(10, 10))
```

#### Step 2: Training the SSOM

After initializing the SSOM, you can train it using your data. For instance, to train it on some sample RGB data:

```python
import numpy as np

# Generate sample data - 100 RGB color points
data = np.random.rand(100, 3)

# Train the SSOM with the data
som.train(data)
```

- **`data`**: A NumPy array where each row represents an input vector (e.g., 100 data points of RGB colors).


### Advanced Features

#### Adjusting Learning Parameters

The SSOM library allows you to adjust the learning rate and neighborhood radius dynamically. For example, if you want to reduce the learning rate over time, you can modify it during the training process:

```python
som.train(data, num_iterations=1000, learning_rate_decay=0.99)
```

- **`learning_rate_decay`**: A factor to reduce the learning rate after each iteration. This helps stabilize the training.

#### Using Custom Distance Functions

The `SSOM` class also provides the ability to specify a custom distance function. By default, it uses Euclidean distance, but you can customize it if needed:

```python
def custom_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))  # Manhattan distance

som = SSOM(grid_size=(10, 10), input_dim=3, distance_fn=custom_distance)
```

### Troubleshooting

- **Training not converging**: Ensure that your learning rate and radius are appropriate for your data size. Lowering the learning rate can help if the SSOM is not converging.
- **Visualization issues**: If the visualization seems cluttered, consider increasing the grid size or using fewer input dimensions.

For further support, refer to the official documentation or open an issue on the library's repository.

### Conclusion

The **SSOM** library is a powerful tool for clustering and visualizing high-dimensional data. By following this guide, you should be able to create, train, and analyze your own SSOM models effectively. For any questions or additional information, please consult the library's documentation or contact the maintainers.



