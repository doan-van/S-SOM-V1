
# SSOM

SSOM is a simple Python library implementing a Self-Organizing Map (SOM) for clustering and visualization.

## Installation

```bash
pip install ssom
```

## Usage

```python
from ssom import SSOM
import numpy as np

# Initialize SSOM
som = SSOM(grid_rows=10, grid_cols=10, num_iterations=100)

# Generate sample data
data = np.random.random((100, 3))

# Train SSOM
mapping = som.train(data)
```
