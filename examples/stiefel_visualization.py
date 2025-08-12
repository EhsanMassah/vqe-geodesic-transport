#!/usr/bin/env python3
"""
Stiefel Manifold Visualization Example

Educational tool for exploring Stiefel manifolds V(n,k) - the set of 
orthogonal k-frames in n-dimensional space.

Usage:
    python examples/stiefel_visualization.py

Features:
- Interactive manifold visualization
- Geodesic path demonstrations  
- Educational geometric insights
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry.run_stiefel import main as run_stiefel_main

def main():
    """Run the Stiefel manifold visualization."""
    print("📐 Stiefel Manifold Visualization")
    print("=" * 35)
    print("Educational tool for exploring differential geometry")
    print("and understanding the geometric foundations of quantum optimization")
    print()
    
    run_stiefel_main()

if __name__ == "__main__":
    main()
