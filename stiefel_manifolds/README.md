# Stiefel Manifolds

This folder contains comprehensive visualizations and utilities for exploring Stiefel manifolds V(n,k) - the space of orthonormal k-frames in Rⁿ.

## Files

### Core Modules
- **`stiefel_manifold.py`** - Main visualization functions for different Stiefel manifolds
- **`run_stiefel.py`** - Interactive runner for individual visualizations
- **`__init__.py`** - Package initialization and exports

## Supported Visualizations

### General Function (New!)
- **`visualize_stiefel_manifold(n, k)`** - Universal function for any V(n,k) with 2 ≤ n ≤ 4, 1 ≤ k ≤ n

### Specific Manifolds
- **V(2,1) = S¹**: Unit circle in R²
- **V(3,1) = S²**: Unit sphere in R³ with random point sampling
- **V(3,2)**: Orthonormal 2-frames in R³, showing vector pairs and Grassmannian projection
- **V(4,1) = S³**: Unit sphere in R⁴ with stereographic projection and cross-sections
- **V(4,2)**: Orthonormal 2-frames in R⁴ with validation checks and projections
- **V(4,3)**: Orthonormal 3-frames in R⁴ with comprehensive analysis

## Usage

### General Function (Recommended)
```python
from stiefel_manifolds import visualize_stiefel_manifold
import matplotlib.pyplot as plt

# Visualize any supported Stiefel manifold
fig = visualize_stiefel_manifold(n=3, k=2, num_samples=200)
plt.show()

# With specific visualization type
fig = visualize_stiefel_manifold(n=4, k=2, visualization_type='validation')
plt.show()
```

### Legacy Individual Functions
```python
from stiefel_manifolds import visualize_stiefel_v31
import matplotlib.pyplot as plt

fig = visualize_stiefel_v31()
plt.show()
```

### Interactive Modes
```bash
cd stiefel_manifolds
python run_stiefel.py          # Menu-driven interface
python stiefel_manifold.py --interactive  # Command-line interactive mode
```

### Complete Demonstration
```bash
cd stiefel_manifolds  
python stiefel_manifold.py     # All examples with general function
```

## Function Parameters

### `visualize_stiefel_manifold(n, k, num_samples=200, visualization_type='auto')`

**Parameters:**
- **n** (int): Ambient space dimension (2 ≤ n ≤ 4)
- **k** (int): Number of orthonormal vectors per frame (1 ≤ k ≤ n)  
- **num_samples** (int): Number of random frames to generate (default: 200)
- **visualization_type** (str): Visualization strategy
  - `'auto'`: Automatic selection based on (n,k)
  - `'geometric'`: Focus on geometric structure
  - `'validation'`: Orthogonality and unit norm checks
  - `'projection'`: High-dimensional projection techniques

**Returns:**
- `matplotlib.Figure`: The generated visualization

**Automatic Strategy Selection:**
- **Unit spheres** (k=1): Direct geometric visualization
- **Low-dimensional frames**: Vector pairs and subspace visualization  
- **High-dimensional frames**: Projections and validation plots
- **4D cases**: Stereographic projections and cross-sections

## Mathematical Background

- **Stiefel Manifold V(n,k)**: Space of orthonormal k-frames in Rⁿ
- **Dimension**: dim(V(n,k)) = nk - k(k+1)/2
- **Special Cases**: 
  - V(n,1) = Sⁿ⁻¹ (unit sphere)
  - V(n,n-1) relates to SO(n)
- **Applications**: Optimization on manifolds, computer vision, robotics

## Technical Details

- Uses QR decomposition for generating uniform random orthonormal frames
- Implements stereographic projection for 4D to 3D visualization
- Includes validation checks for orthogonality and unit norms
- Connects to Grassmannian manifolds for subspace visualization
