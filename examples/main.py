#!/usr/bin/env python3
"""
VQE Geodesic Transport - Main Examples

Choose between molecular VQE calculations or geometric visualizations.
This is the main entry point for exploring the project capabilities.
"""

import sys
import os

def main():
    """Main menu for project examples."""
    print("🚀 VQE Geodesic Transport - Enhanced Performance")
    print("=" * 50)
    print("Choose an example to run:")
    print()
    print("1. 🧬 Molecular VQE (H2, LiH, BeH2) - 200x speedup!")
    print("2. 📐 Stiefel Manifold Visualization") 
    print("3. ⚡ Quick H2 test (recommended first run)")
    print("4. 📊 Performance comparison")
    print("0. Exit")
    print()
    
    try:
        choice = input("Enter your choice (0-4): ").strip()
        
        if choice == "1":
            print("\nLaunching Enhanced Molecular VQE...")
            os.system(f"cd '{os.path.dirname(__file__)}/..' && python -m src.quantum_vqe.qibo_molecular_vqe")
            
        elif choice == "2":
            print("\nLaunching Stiefel Manifold Visualization...")
            os.system(f"cd '{os.path.dirname(__file__)}/..' && python -m src.geometry.run_stiefel")
            
        elif choice == "3":
            print("\nRunning Quick H2 Test (EGT-optimized ansatz)...")
            cmd = f"cd '{os.path.dirname(__file__)}/..' && echo -e '1\\n1\\n4\\nn\\ny' | python -m src.quantum_vqe.qibo_molecular_vqe"
            os.system(cmd)
            
        elif choice == "4":
            print("\nPerformance Comparison:")
            print("=" * 30)
            print("📈 Enhanced EGT (v2.0):  0.7s runtime, 3.96% error")
            print("📉 Original method:     130s runtime, 12.14% error") 
            print("🎯 Speedup achieved:    200x faster, 3x more accurate")
            print("\nFor detailed analysis, see docs/performance_analysis.md")
            
        elif choice == "0":
            print("Goodbye!")
            return
            
        else:
            print("Invalid choice. Please try again.")
            main()
            
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
