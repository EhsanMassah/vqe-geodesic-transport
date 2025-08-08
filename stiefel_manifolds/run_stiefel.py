#!/usr/bin/env python3
"""
Stiefel Manifold Interactive Runner

Quick runner for individual Stiefel manifold visualizations using the general function.
"""

import sys
from stiefel_manifold import (
    visualize_stiefel_manifold,
    demo_interactive
)
import matplotlib.pyplot as plt


def show_menu():
    """Display visualization menu"""
    print("\n🔺 Stiefel Manifold Visualizations")
    print("=" * 45)
    print("1. V(2,1) = S¹ - Unit circle in R²")
    print("2. V(3,1) = S² - Unit sphere in R³")
    print("3. V(3,2) - Orthonormal 2-frames in R³")  
    print("4. V(4,1) = S³ - Unit sphere in R⁴")
    print("5. V(4,2) - Orthonormal 2-frames in R⁴")
    print("6. V(4,3) - Orthonormal 3-frames in R⁴")
    print("7. Custom V(n,k) - Enter your own parameters")
    print("8. Interactive mode")
    print("9. All examples")
    print("0. Exit")
    print("=" * 45)


def main():
    """Interactive runner for Stiefel manifold visualizations"""
    
    # Predefined examples
    examples = {
        1: (2, 1, "V(2,1) = S¹"),
        2: (3, 1, "V(3,1) = S²"),
        3: (3, 2, "V(3,2)"),
        4: (4, 1, "V(4,1) = S³"),
        5: (4, 2, "V(4,2)"),
        6: (4, 3, "V(4,3)")
    }
    
    while True:
        show_menu()
        
        try:
            choice = input("\nSelect option (0-9): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
                
            elif choice in ['1', '2', '3', '4', '5', '6']:
                choice_num = int(choice)
                n, k, desc = examples[choice_num]
                
                print(f"\n🔹 Generating {desc} visualization...")
                
                # Calculate and display dimension
                manifold_dim = n * k - k * (k + 1) // 2
                print(f"   Manifold dimension: {manifold_dim}")
                
                fig = visualize_stiefel_manifold(n, k, num_samples=200)
                plt.show()
                
            elif choice == '7':
                print("\n🔹 Custom V(n,k) visualization")
                print("Valid ranges: 2 ≤ n ≤ 4, 1 ≤ k ≤ n")
                
                try:
                    n = int(input("Enter ambient dimension n: "))
                    k = int(input("Enter frame size k: "))
                    
                    print(f"\n🔹 Generating V({n},{k}) visualization...")
                    
                    manifold_dim = n * k - k * (k + 1) // 2
                    print(f"   Manifold dimension: {manifold_dim}")
                    
                    fig = visualize_stiefel_manifold(n, k, num_samples=200)
                    plt.show()
                    
                except ValueError:
                    print("❌ Invalid input. Please enter integers.")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
            elif choice == '8':
                demo_interactive()
                
            elif choice == '9':
                print("\n🔹 Generating all example visualizations...")
                
                for i, (n, k, desc) in examples.items():
                    print(f"\n{i}/6: {desc}")
                    manifold_dim = n * k - k * (k + 1) // 2
                    print(f"     Dimension: {manifold_dim}")
                    
                    try:
                        fig = visualize_stiefel_manifold(n, k, num_samples=150)
                        plt.show()
                    except Exception as e:
                        print(f"     ❌ Error: {e}")
                
                print("\n✅ All visualizations complete!")
                
            else:
                print("❌ Invalid choice. Please select 0-9.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
