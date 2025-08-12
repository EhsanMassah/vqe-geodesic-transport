#!/usr/bin/env python3
"""
Interactive runner for Quantum Geodesic Transport VQA

Provides a menu-driven interface to explore variational quantum algorithms
with geodesic transport optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_geodesic_vqa import (
    QuantumVariationalEigensolver, create_test_hamiltonian,
    analyze_quantum_geometry, demonstrate_geodesic_vqe
)


def menu_main():
    """Main menu for quantum geodesic transport demonstrations"""
    
    while True:
        print("\n" + "="*60)
        print("🌟 Quantum Geodesic Transport VQA - Interactive Menu")
        print("="*60)
        print("1. Run VQE with Geodesic Transport")
        print("2. Analyze Quantum Parameter Manifold Geometry")
        print("3. Compare Optimization Methods")
        print("4. Custom Hamiltonian VQE")
        print("5. Full Demonstration")
        print("6. Help - About the Methods")
        print("0. Exit")
        print("-"*60)
        
        choice = input("Select option (0-6): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            run_single_vqe()
        elif choice == '2':
            analyze_quantum_geometry()
        elif choice == '3':
            compare_optimization_methods()
        elif choice == '4':
            custom_hamiltonian_vqe()
        elif choice == '5':
            demonstrate_geodesic_vqe()
        elif choice == '6':
            show_help()
        else:
            print("❌ Invalid choice. Please select 0-6.")


def run_single_vqe():
    """Run a single VQE optimization with user parameters"""
    
    print("\n🔬 Single VQE Run with Geodesic Transport")
    print("-"*45)
    
    try:
        # Get user inputs
        n_qubits = int(input("Number of qubits (2-4 recommended): ") or "2")
        
        print("\nHamiltonian types:")
        print("1. Ising model (default)")
        print("2. Heisenberg model")
        print("3. Random Hermitian")
        
        ham_choice = input("Select Hamiltonian type (1-3): ").strip() or "1"
        ham_types = {"1": "ising", "2": "heisenberg", "3": "random"}
        ham_type = ham_types.get(ham_choice, "ising")
        
        n_layers = int(input("Number of circuit layers (1-3): ") or "2")
        n_iterations = int(input("Number of optimization iterations (10-100): ") or "50")
        learning_rate = float(input("Learning rate (0.001-0.1): ") or "0.05")
        
        print(f"\nRunning VQE...")
        print(f"• {n_qubits} qubits, {n_layers} layers")
        print(f"• {ham_type} Hamiltonian")
        print(f"• {n_iterations} iterations, lr={learning_rate}")
        
        # Create Hamiltonian and run VQE
        H = create_test_hamiltonian(n_qubits, ham_type)
        vqe = QuantumVariationalEigensolver(H, n_qubits, n_layers)
        
        result = vqe.find_ground_state(
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            random_seed=42
        )
        
        # Plot results
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(result['energy_history'], 'b-', linewidth=2)
        if result['exact_ground_energy'] is not None:
            plt.axhline(y=result['exact_ground_energy'], color='r', 
                       linestyle='--', label=f"Exact: {result['exact_ground_energy']:.6f}")
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('VQE Convergence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        state_probs = np.abs(result['final_state'])**2
        plt.bar(range(len(state_probs)), state_probs, alpha=0.7)
        plt.xlabel('Basis State')
        plt.ylabel('Probability')
        plt.title('Final Quantum State')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def compare_optimization_methods():
    """Compare geodesic transport with standard gradient descent"""
    
    print("\n📊 Optimization Method Comparison")
    print("-"*40)
    
    # Create test system
    n_qubits = 2
    H = create_test_hamiltonian(n_qubits, "ising")
    
    print(f"Comparing optimization methods on {n_qubits}-qubit Ising model...")
    
    # Run geodesic VQE
    print("\n1. Geodesic Transport VQE:")
    vqe_geodesic = QuantumVariationalEigensolver(H, n_qubits, 2)
    result_geodesic = vqe_geodesic.find_ground_state(
        n_iterations=30, learning_rate=0.05, random_seed=42
    )
    
    # Simple gradient descent (simplified implementation)
    print("\n2. Standard Gradient Descent:")
    from quantum_geodesic_vqa import QuantumParameterManifold, QuantumCircuit
    
    circuit = QuantumCircuit(n_qubits, 2)
    manifold = QuantumParameterManifold(circuit, H)
    
    # Initialize same parameters
    np.random.seed(42)
    params = np.random.uniform(-np.pi, np.pi, circuit.n_params)
    energy_history_standard = []
    
    for i in range(30):
        energy = manifold.energy_expectation(params)
        energy_history_standard.append(energy)
        
        if i % 10 == 0:
            print(f"Iteration {i:2d}: Energy = {energy:.8f}")
        
        # Standard gradient step
        gradient = manifold.energy_gradient(params)
        params -= 0.05 * gradient  # Simple gradient descent
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(result_geodesic['energy_history'], 'b-', linewidth=2, label='Geodesic Transport')
    plt.plot(energy_history_standard, 'r--', linewidth=2, label='Standard Gradient')
    
    if result_geodesic['exact_ground_energy'] is not None:
        plt.axhline(y=result_geodesic['exact_ground_energy'], color='g', 
                   linestyle=':', alpha=0.7, label='Exact Ground State')
    
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Optimization Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    final_energies = [result_geodesic['energy_history'][-1], energy_history_standard[-1]]
    methods = ['Geodesic\nTransport', 'Standard\nGradient']
    colors = ['blue', 'red']
    
    bars = plt.bar(methods, final_energies, color=colors, alpha=0.7)
    plt.ylabel('Final Energy')
    plt.title('Final Optimization Results')
    
    # Add value labels on bars
    for bar, energy in zip(bars, final_energies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{energy:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nComparison Results:")
    print(f"• Geodesic Transport final energy: {result_geodesic['energy_history'][-1]:.6f}")
    print(f"• Standard Gradient final energy: {energy_history_standard[-1]:.6f}")
    if result_geodesic['exact_ground_energy'] is not None:
        print(f"• Exact ground state energy: {result_geodesic['exact_ground_energy']:.6f}")
        print(f"• Geodesic error: {abs(result_geodesic['energy_history'][-1] - result_geodesic['exact_ground_energy']):.6f}")
        print(f"• Standard error: {abs(energy_history_standard[-1] - result_geodesic['exact_ground_energy']):.6f}")


def custom_hamiltonian_vqe():
    """Allow user to define custom Hamiltonian matrices"""
    
    print("\n🔧 Custom Hamiltonian VQE")
    print("-"*30)
    
    print("Options for custom Hamiltonians:")
    print("1. 2x2 Pauli Hamiltonian (1 qubit)")
    print("2. 4x4 Two-qubit Hamiltonian")
    print("3. Random symmetric matrix")
    
    choice = input("Select option (1-3): ").strip() or "1"
    
    if choice == "1":
        # Single qubit Pauli Hamiltonian: H = a*X + b*Y + c*Z
        print("\nSingle qubit Hamiltonian: H = a*X + b*Y + c*Z")
        a = float(input("Coefficient for X (default 0.5): ") or "0.5")
        b = float(input("Coefficient for Y (default 0.0): ") or "0.0")
        c = float(input("Coefficient for Z (default 1.0): ") or "1.0")
        
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        H = a * X + b * Y + c * Z
        n_qubits = 1
        
    elif choice == "2":
        print("\nTwo-qubit Hamiltonian builder")
        print("H = a*XX + b*YY + c*ZZ + d*XZ + e*ZX")
        a = float(input("Coefficient for XX (default 1.0): ") or "1.0")
        b = float(input("Coefficient for YY (default 1.0): ") or "1.0") 
        c = float(input("Coefficient for ZZ (default 1.0): ") or "1.0")
        d = float(input("Coefficient for XZ (default 0.0): ") or "0.0")
        e = float(input("Coefficient for ZX (default 0.0): ") or "0.0")
        
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        ZZ = np.kron(Z, Z)
        XZ = np.kron(X, Z)
        ZX = np.kron(Z, X)
        
        H = a * XX + b * YY + c * ZZ + d * XZ + e * ZX
        n_qubits = 2
        
    else:  # Random matrix
        size = int(input("Matrix size (power of 2, default 4): ") or "4")
        if size & (size - 1) != 0:  # Check if power of 2
            size = 4
        n_qubits = int(np.log2(size))
        
        np.random.seed(42)
        H = np.random.randn(size, size)
        H = (H + H.T) / 2  # Make symmetric
    
    print(f"\nHamiltonian created:")
    print(f"Size: {H.shape}")
    print(f"Eigenvalues: {np.linalg.eigvals(H)}")
    
    # Run VQE
    n_layers = int(input(f"Number of layers for {n_qubits}-qubit circuit (default 2): ") or "2")
    
    vqe = QuantumVariationalEigensolver(H, n_qubits, n_layers)
    result = vqe.find_ground_state(n_iterations=50, learning_rate=0.05)
    
    # Visualize Hamiltonian and results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot Hamiltonian matrix
    im = axes[0].imshow(np.real(H), cmap='RdBu', interpolation='nearest')
    axes[0].set_title('Hamiltonian Matrix (Real Part)')
    axes[0].set_xlabel('Matrix Index')
    axes[0].set_ylabel('Matrix Index')
    plt.colorbar(im, ax=axes[0])
    
    # Plot eigenvalue spectrum
    eigenvals = np.sort(np.linalg.eigvals(H))
    axes[1].bar(range(len(eigenvals)), eigenvals, alpha=0.7)
    axes[1].axhline(y=result['final_energy'], color='r', linestyle='--', 
                   label=f'VQE: {result["final_energy"]:.4f}')
    axes[1].set_xlabel('Eigenvalue Index')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Hamiltonian Spectrum')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot convergence
    axes[2].plot(result['energy_history'], 'b-', linewidth=2)
    axes[2].axhline(y=np.min(eigenvals), color='g', linestyle=':', 
                   label=f'Exact: {np.min(eigenvals):.4f}')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('VQE Convergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def show_help():
    """Display help information about the methods"""
    
    print("\n📚 Help - Quantum Geodesic Transport VQA")
    print("="*50)
    
    help_text = """
🌟 OVERVIEW:
This implementation demonstrates Variational Quantum Algorithms (VQA) enhanced 
with exact geodesic transport on quantum parameter manifolds.

🔬 KEY CONCEPTS:

1. QUANTUM PARAMETER MANIFOLDS:
   • The space of quantum circuit parameters forms a Riemannian manifold
   • The metric is given by the Quantum Fisher Information Matrix (QFIM)
   • QFIM captures the geometric structure of the quantum state space

2. GEODESIC TRANSPORT:
   • Natural way to move on curved manifolds following shortest paths
   • Preserves geometric properties during optimization
   • More efficient than standard gradient descent on curved spaces

3. NATURAL GRADIENT DESCENT:
   • Uses the inverse QFIM to precondition gradients
   • Gradient direction: ∇̃f = G⁻¹∇f (G = QFIM)
   • Follows the steepest descent in the manifold's natural metric

4. VARIATIONAL QUANTUM EIGENSOLVER (VQE):
   • Finds ground states of quantum Hamiltonians
   • Uses parameterized quantum circuits as ansatz
   • Optimizes expectation value ⟨ψ(θ)|H|ψ(θ)⟩

🔧 COMPUTATIONAL METHODS:

• Quantum Fisher Information Matrix computation
• Christoffel symbols for geodesic equations
• Parallel transport along geodesics
• Exponential map for geodesic updates
• Natural gradient optimization

📊 VISUALIZATIONS:

• Energy convergence plots
• Quantum state probability distributions
• QFIM eigenvalue analysis
• Manifold geometry visualization
• Optimization method comparisons

🎯 APPLICATIONS:

• Quantum chemistry (molecular ground states)
• Condensed matter physics (spin systems)
• Quantum optimization problems
• Machine learning on quantum manifolds

📖 REFERENCES:

• arXiv:2506.17395v2 - "Variational quantum algorithms with exact geodesic transport"
• Quantum Fisher Information and Natural Gradients
• Riemannian Optimization for Quantum Computing
• Variational Quantum Algorithms Review
"""
    
    print(help_text)
    input("\nPress Enter to continue...")


if __name__ == "__main__":
    menu_main()
