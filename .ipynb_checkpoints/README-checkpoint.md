# Structural Self Organizing Map Library - Tutorial

Welcome to the **S-SOM** (Structural Self-organizing Map) library! 

**S-SOM** stands for **Structural Self-organizing Map**. SSOM is introduced by Doan et al. (2021) primarily to enhance the ability of SOM in weather pattern detection. 

Similar with SOM, S-SOM is a type of artificial neural network used for clustering and visualizing high-dimensional data. The SSOM algorithm projects input data into a limited "topology" space, typically in the form of a map-like 2-D grid, or ring-like 1-D grid, which reveals the relationships and structure of input data. The difference between S-SOM and "conventional" SOM is the ability of S-SOM in detect the similarity between "structured" data (i.e., data which having time or space orders) through utilizing "structural" similarity (instead of simple Euclidean distance) when search the best matching unit.

The main class in this library is `SSOM`, which allows you to:
- Create a new SSOM instance.
- Train the SSOM with data.

This guide will help you understand how to install and use the library, including detailed examples of how to create and manipulate SSOMs.

Please cite the below paper if you are using this library or source code provided withit it.

***
Doan, Q.-V., Kusaka, H., Sato, T., and Chen, F.: S-SOM v1.0: a structural self-organizing map algorithm for weather typing, Geosci. Model Dev., 14, 2097–2111, https://doi.org/10.5194/gmd-14-2097-2021, 2021.
***


## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Advanced Features](#advanced-features)
4. [Troubleshooting](#troubleshooting)


### Installation

To use the SSOM library, ensure you have Python 3.x installed along with the following dependencies:

```bash
pip install numpy matplotlib
```

**`There are three methods to install or use the library`**

`1. Install directly from this Github account`

```bash
pip install git+https://github.com/doan-van/S-SOM-V1.git
```
* Note: This can work only if you have **`git`** alread-installed in your PC.

`2. Download source code and install it from local directory.`


```bash
git clone https://github.com/doan-van/S-SOM-V1.git
```
* Note: This can work only if you have **`git`** alread-installed in your PC.
* Otherwise, you have to download whole package manually and extract it to your favorite place in your PC.

Navigate to the cloned directory and install the library locally:

```bash
cd S-SOM-V1
pip install .
```
`3. Alternatively, download the `ssom.py` file from this GitHub repository and add it to your working directory.`


### Getting Started

`Check ` **test/** `for more examples of how to run and visualize SSOM`

#### Step 1: Creating and Initializing the SSOM

To get started with the SSOM library, import the `SSOM` class and create an instance as shown below:

```python
from ssom import SSOM

# Define the SSOM parameters
som = SSOM(grid_size=(10, 10))
```

- **`grid_size`**: Tuple representing the dimensions of the grid (e.g., 10x10).
- default implementation uses Euclidean distance for searching BMU, there option `metric="ssim"` for using Structural Similarity to define BMU.

#### Step 2: Training the SSOM

After initializing the SSOM, you can train it using input data. 

```python
som.train(data)
```

- **`data`**: A NumPy array where each row represents an input vector.


### Advanced Features

#### Adjusting Learning Parameters

The SSOM library allows you to adjust the learning rate and neighborhood radius dynamically. For example, if you want to reduce the learning rate over time, you can modify it during the training process:

```python
som.train(data, num_iterations=1000, learning_rate_decay=0.99)
```

- **`learning_rate_decay`**: A factor to reduce the learning rate after each iteration. This helps stabilize the training.

#### Using Custom Distance Functions

The `SSOM` class also provides the ability to specify a custom distance function. By default, it uses Euclidean distance, but one can customize by selecting option `metric="ssim"` for using Structural Similarity to define BMU.


### Troubleshooting

- **Training not converging**: Ensure that your learning rate and radius are appropriate for your data size. Lowering the learning rate can help if the SSOM is not converging.





