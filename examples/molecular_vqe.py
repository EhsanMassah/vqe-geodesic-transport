#!/usr/bin/env python3
"""
Enhanced Molecular VQE Example

This is the main entry point for running molecular VQE calculations
with geodesic transport optimization.

Usage:
    python examples/main.py

Features:
- Interactive molecule selection (H2, LiH, BeH2)
- Multiple ansatz options including EGT-optimized
- Fast approximation algorithms (200x speedup)
- Comprehensive analysis and visualization
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_vqe.qibo_molecular_vqe import ElectronicStructureVQE

def main():
    """Run the enhanced molecular VQE interface."""
    print("🧬 Enhanced Molecular VQE with Geodesic Transport")
    print("=" * 50)
    print("Features:")
    print("• 200x speed improvement with fast approximations")
    print("• EGT-optimized ansatz from arXiv:2506.17395v2") 
    print("• Interactive molecule and circuit selection")
    print("• Comprehensive analysis and visualization")
    print()
    
    # Create and run the VQE interface
    vqe = ElectronicStructureVQE()
    vqe.run_interactive()

if __name__ == "__main__":
    main()
