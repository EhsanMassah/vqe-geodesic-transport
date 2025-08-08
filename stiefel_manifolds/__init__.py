"""
Stiefel Manifolds Package

This package contains visualizations and utilities for exploring Stiefel manifolds.

Modules:
--------
- stiefel_manifold: Core visualization functions for V(n,k) manifolds
- run_stiefel: Interactive runner for individual visualizations

Usage:
------
```python
from stiefel_manifolds.stiefel_manifold import visualize_stiefel_v31
import matplotlib.pyplot as plt

fig = visualize_stiefel_v31()
plt.show()
```

Or run interactively:
```bash
cd stiefel_manifolds
python run_stiefel.py
```
"""

from .stiefel_manifold import (
    generate_random_orthonormal_frame,
    visualize_stiefel_manifold,
    visualize_stiefel_v31,
    visualize_stiefel_v32,
    visualize_stiefel_v41,
    visualize_stiefel_v42,
    demo_interactive
)

__version__ = "2.0.0"
__author__ = "Geometry Workspace"
__all__ = [
    "generate_random_orthonormal_frame",
    "visualize_stiefel_manifold",
    "visualize_stiefel_v31",
    "visualize_stiefel_v32", 
    "visualize_stiefel_v41",
    "visualize_stiefel_v42",
    "demo_interactive"
]
