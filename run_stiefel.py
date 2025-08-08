#!/usr/bin/env python3
"""
Stiefel Manifold Interactive Runner

Quick runner for individual Stiefel manifold visualizations.
"""

import sys
from stiefel_manifold import (
    visualize_stiefel_v31, 
    visualize_stiefel_v32, 
    visualize_stiefel_v41, 
    visualize_stiefel_v42
)
import matplotlib.pyplot as plt


def show_menu():
    """Display visualization menu"""
    print("\n🔺 Stiefel Manifold Visualizations")
    print("=" * 40)
    print("1. V(3,1) = S² - Unit sphere in R³")
    print("2. V(3,2) - Orthonormal 2-frames in R³")  
    print("3. V(4,1) = S³ - Unit sphere in R⁴")
    print("4. V(4,2) - Orthonormal 2-frames in R⁴")
    print("5. All visualizations")
    print("0. Exit")
    print("=" * 40)


def main():
    """Interactive runner for Stiefel manifold visualizations"""
    
    while True:
        show_menu()
        
        try:
            choice = input("\nSelect visualization (0-5): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
                
            elif choice == '1':
                print("\n🔹 Generating V(3,1) = S² visualization...")
                fig = visualize_stiefel_v31()
                plt.show()
                
            elif choice == '2':
                print("\n🔹 Generating V(3,2) visualization...")
                fig = visualize_stiefel_v32()
                plt.show()
                
            elif choice == '3':
                print("\n🔹 Generating V(4,1) = S³ visualization...")
                fig = visualize_stiefel_v41()
                plt.show()
                
            elif choice == '4':
                print("\n🔹 Generating V(4,2) visualization...")
                fig = visualize_stiefel_v42()
                plt.show()
                
            elif choice == '5':
                print("\n🔹 Generating all visualizations...")
                
                print("1/4: V(3,1) = S²")
                fig1 = visualize_stiefel_v31()
                plt.show()
                
                print("2/4: V(3,2)")
                fig2 = visualize_stiefel_v32()
                plt.show()
                
                print("3/4: V(4,1) = S³")
                fig3 = visualize_stiefel_v41()
                plt.show()
                
                print("4/4: V(4,2)")
                fig4 = visualize_stiefel_v42()
                plt.show()
                
                print("✅ All visualizations complete!")
                
            else:
                print("❌ Invalid choice. Please select 0-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
