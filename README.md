# Advanced Geometry Experimental Workspace

A minimal Python workspace for experimental advanced geometry computations and visualizations.

## Overview

This workspace is set up for:
- Geometric algorithm development and testing
- Mathematical computations with NumPy
- 2D/3D geometric visualizations with Matplotlib
- Research and experimentation in computational geometry

## Structure

- `main.py` - Main entry point for geometry experiments
- `stiefel_manifolds/` - Package for Stiefel manifold visualizations
  - `stiefel_manifold.py` - Core visualization functions
  - `run_stiefel.py` - Interactive runner
  - `README.md` - Package documentation
- `.github/copilot-instructions.md` - Custom instructions for better code assistance

## Getting Started

1. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib scipy
   ```

3. **Run the example:**
   ```bash
   python main.py
   ```

4. **Explore Stiefel manifolds:**
   ```bash
   cd stiefel_manifolds
   python stiefel_manifold.py    # All visualizations
   python run_stiefel.py         # Interactive menu
   ```

## Dependencies

- **NumPy**: Numerical computing and array operations
- **Matplotlib**: 2D plotting and visualization
- **SciPy**: Advanced mathematical functions and algorithms

## Usage

The workspace is intentionally minimal to allow for flexible experimentation. Start by modifying `main.py` or creating new Python files for your geometric algorithms and visualizations.

## Examples

The main script includes a simple unit circle visualization to verify the setup is working correctly.
