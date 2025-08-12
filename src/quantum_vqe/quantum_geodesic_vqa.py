#!/usr/bin/env python3
"""
Variational Quantum Algorithms with Exact Geodesic Transport

Implementation of quantum variational algorithms enhanced with geodesic transport
techniques for optimization on quantum parameter manifolds.

Based on arXiv:2506.17395v2 - "Variational quantum algorithms with exact geodesic transport"

Key concepts implemented:
1. Quantum parameter manifolds and Riemannian geometry
2. Geodesic transport for parameter updates
3. Natural gradient descent on quantum manifolds
4. Quantum Fisher Information Metric (QFIM)
5. Variational quantum eigensolver (VQE) with geodesic optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class QuantumCircuit:
    """Simple quantum circuit representation for VQA"""
    
    def __init__(self, n_qubits: int, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = self._count_parameters()
        
    def _count_parameters(self) -> int:
        """Count total number of parameters in the ansatz"""
        # Each layer has n_qubits rotation gates + entangling gates
        params_per_layer = 3 * self.n_qubits  # RX, RY, RZ for each qubit
        return params_per_layer * self.n_layers
    
    def state_vector(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute the quantum state vector for given parameters
        
        This is a simplified representation - in practice would use
        quantum circuit simulation libraries like Qiskit, Cirq, or PennyLane
        """
        # Initialize |0...0⟩ state
        state = np.zeros(2**self.n_qubits, dtype=complex)
        state[0] = 1.0
        
        # Apply parameterized layers (simplified representation)
        for layer in range(self.n_layers):
            layer_params = parameters[layer * 3 * self.n_qubits:(layer + 1) * 3 * self.n_qubits]
            state = self._apply_layer(state, layer_params)
            
        return state
    
    def _apply_layer(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply a single parameterized layer"""
        # Simplified: just rotate the state vector using parameters
        # In practice, this would be proper quantum gate operations
        
        # Create rotation matrix based on parameters
        n = len(params)
        rotation_matrix = np.eye(len(state), dtype=complex)
        
        # Apply parameterized rotations (simplified)
        for i in range(0, n, 3):
            if i + 2 < n:
                theta_x, theta_y, theta_z = params[i:i+3]
                # Simplified rotation effect
                phase = np.exp(1j * (theta_x + theta_y + theta_z) / 3)
                rotation_matrix *= phase
                
        return rotation_matrix @ state


class QuantumParameterManifold:
    """
    Riemannian manifold structure for quantum circuit parameters
    
    Implements the quantum Fisher Information Metric (QFIM) and
    geodesic computations for parameter optimization.
    """
    
    def __init__(self, circuit: QuantumCircuit, hamiltonian: np.ndarray):
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.n_params = circuit.n_params
        
    def quantum_fisher_information_matrix(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute the Quantum Fisher Information Matrix (QFIM)
        
        The QFIM provides the Riemannian metric on the quantum parameter manifold
        """
        qfim = np.zeros((self.n_params, self.n_params))
        
        # Get the quantum state and its derivatives
        state = self.circuit.state_vector(parameters)
        state_derivatives = self._compute_state_derivatives(parameters)
        
        for i in range(self.n_params):
            for j in range(self.n_params):
                # QFIM elements: G_ij = 4 * Re[⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩]
                overlap_ij = np.vdot(state_derivatives[i], state_derivatives[j])
                overlap_i = np.vdot(state_derivatives[i], state)
                overlap_j = np.vdot(state, state_derivatives[j])
                
                qfim[i, j] = 4 * np.real(overlap_ij - overlap_i * overlap_j)
        
        return qfim
    
    def _compute_state_derivatives(self, parameters: np.ndarray, 
                                  eps: float = 1e-8) -> List[np.ndarray]:
        """Compute derivatives of quantum state with respect to parameters"""
        derivatives = []
        
        for i in range(self.n_params):
            # Finite difference approximation
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            state_plus = self.circuit.state_vector(params_plus)
            state_minus = self.circuit.state_vector(params_minus)
            
            derivative = (state_plus - state_minus) / (2 * eps)
            derivatives.append(derivative)
            
        return derivatives
    
    def energy_expectation(self, parameters: np.ndarray) -> float:
        """Compute energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩"""
        state = self.circuit.state_vector(parameters)
        return np.real(np.vdot(state, self.hamiltonian @ state))
    
    def energy_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Compute gradient of energy expectation value"""
        gradient = np.zeros(self.n_params)
        eps = 1e-8
        
        for i in range(self.n_params):
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            energy_plus = self.energy_expectation(params_plus)
            energy_minus = self.energy_expectation(params_minus)
            
            gradient[i] = (energy_plus - energy_minus) / (2 * eps)
            
        return gradient


class GeodesicTransport:
    """
    Geodesic transport operations on quantum parameter manifolds
    
    Implements parallel transport and geodesic computation using
    the quantum Fisher Information metric.
    """
    
    def __init__(self, manifold: QuantumParameterManifold):
        self.manifold = manifold
        
    def parallel_transport(self, vector: np.ndarray, 
                          start_point: np.ndarray, 
                          end_point: np.ndarray,
                          n_steps: int = 10) -> np.ndarray:
        """
        Parallel transport a vector along a geodesic
        
        Parameters:
        -----------
        vector : np.ndarray
            Vector to transport
        start_point : np.ndarray
            Starting point on manifold
        end_point : np.ndarray
            End point on manifold
        n_steps : int
            Number of discretization steps
            
        Returns:
        --------
        np.ndarray
            Transported vector at end_point
        """
        # Compute geodesic path
        path = self._compute_geodesic_path(start_point, end_point, n_steps)
        
        # Transport vector along the path
        transported_vector = vector.copy()
        
        for i in range(n_steps - 1):
            current_point = path[i]
            next_point = path[i + 1]
            
            # Compute connection coefficients (Christoffel symbols)
            christoffel = self._compute_christoffel_symbols(current_point)
            
            # Parallel transport update
            tangent = next_point - current_point
            dt = 1.0 / n_steps
            
            # Transport equation: dV/dt + Γ(V, T) = 0
            transport_correction = np.zeros_like(transported_vector)
            for j in range(len(transported_vector)):
                for k in range(len(tangent)):
                    for l in range(len(transported_vector)):
                        transport_correction[j] += (christoffel[j, k, l] * 
                                                   transported_vector[l] * 
                                                   tangent[k])
            
            transported_vector -= dt * transport_correction
            
        return transported_vector
    
    def _compute_geodesic_path(self, start: np.ndarray, end: np.ndarray, 
                              n_steps: int) -> np.ndarray:
        """
        Compute geodesic path between two points
        
        Uses the geodesic equation: d²x/dt² + Γ(dx/dt, dx/dt) = 0
        """
        path = np.zeros((n_steps, len(start)))
        path[0] = start
        path[-1] = end
        
        # Initial velocity (straight line approximation)
        velocity = (end - start) / (n_steps - 1)
        
        # Integrate geodesic equation
        dt = 1.0 / (n_steps - 1)
        
        for i in range(1, n_steps - 1):
            current_pos = path[i - 1]
            
            # Compute Christoffel symbols at current position
            christoffel = self._compute_christoffel_symbols(current_pos)
            
            # Geodesic acceleration
            acceleration = np.zeros_like(velocity)
            for j in range(len(velocity)):
                for k in range(len(velocity)):
                    for l in range(len(velocity)):
                        acceleration[j] -= christoffel[j, k, l] * velocity[k] * velocity[l]
            
            # Update position and velocity
            path[i] = current_pos + velocity * dt + 0.5 * acceleration * dt**2
            velocity += acceleration * dt
            
        return path
    
    def _compute_christoffel_symbols(self, point: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols Γᵢⱼₖ at a given point
        
        Γᵢⱼₖ = 0.5 * gⁱˡ * (∂gⱼˡ/∂xₖ + ∂gₖˡ/∂xⱼ - ∂gⱼₖ/∂xˡ)
        """
        n = len(point)
        christoffel = np.zeros((n, n, n))
        eps = 1e-6
        
        # Get metric tensor and its inverse
        metric = self.manifold.quantum_fisher_information_matrix(point)
        
        # Regularize metric for numerical stability
        metric += 1e-10 * np.eye(n)
        
        try:
            metric_inv = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if metric is singular
            metric_inv = np.linalg.pinv(metric)
        
        # Compute metric derivatives
        metric_derivatives = np.zeros((n, n, n))
        
        for k in range(n):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[k] += eps
            point_minus[k] -= eps
            
            metric_plus = self.manifold.quantum_fisher_information_matrix(point_plus)
            metric_minus = self.manifold.quantum_fisher_information_matrix(point_minus)
            
            metric_derivatives[:, :, k] = (metric_plus - metric_minus) / (2 * eps)
        
        # Compute Christoffel symbols
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        christoffel[i, j, k] += 0.5 * metric_inv[i, l] * (
                            metric_derivatives[j, l, k] + 
                            metric_derivatives[k, l, j] - 
                            metric_derivatives[j, k, l]
                        )
        
        return christoffel
    
    def exponential_map(self, point: np.ndarray, tangent: np.ndarray, 
                       t: float = 1.0) -> np.ndarray:
        """
        Exponential map: exp_p(v) - follows geodesic from p in direction v
        """
        n_steps = max(10, int(50 * t))
        
        # Initialize geodesic
        current_pos = point.copy()
        velocity = tangent.copy()
        dt = t / n_steps
        
        # Integrate geodesic equation
        for i in range(n_steps):
            christoffel = self._compute_christoffel_symbols(current_pos)
            
            # Compute acceleration
            acceleration = np.zeros_like(velocity)
            for j in range(len(velocity)):
                for k in range(len(velocity)):
                    for l in range(len(velocity)):
                        acceleration[j] -= christoffel[j, k, l] * velocity[k] * velocity[l]
            
            # Update position and velocity
            current_pos += velocity * dt + 0.5 * acceleration * dt**2
            velocity += acceleration * dt
            
        return current_pos


class NaturalGradientOptimizer:
    """
    Natural gradient optimizer using the quantum Fisher Information metric
    
    Implements geodesic optimization for variational quantum algorithms
    """
    
    def __init__(self, manifold: QuantumParameterManifold, 
                 geodesic_transport: GeodesicTransport):
        self.manifold = manifold
        self.geodesic_transport = geodesic_transport
        
    def natural_gradient_step(self, parameters: np.ndarray, 
                             learning_rate: float = 0.01) -> np.ndarray:
        """
        Perform one step of natural gradient descent
        
        Natural gradient: ∇̃f = G⁻¹∇f, where G is the QFIM
        """
        # Compute standard gradient
        gradient = self.manifold.energy_gradient(parameters)
        
        # Compute QFIM (metric tensor)
        qfim = self.manifold.quantum_fisher_information_matrix(parameters)
        
        # Regularize for numerical stability
        n = len(parameters)
        qfim += 1e-8 * np.eye(n)
        
        try:
            # Natural gradient: G⁻¹∇f
            natural_gradient = np.linalg.solve(qfim, gradient)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            natural_gradient = np.linalg.pinv(qfim) @ gradient
        
        # Update parameters using geodesic step
        new_parameters = self.geodesic_transport.exponential_map(
            parameters, -learning_rate * natural_gradient
        )
        
        return new_parameters
    
    def optimize(self, initial_parameters: np.ndarray, 
                n_iterations: int = 100,
                learning_rate: float = 0.01,
                tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """
        Run natural gradient optimization
        
        Returns:
        --------
        Tuple[np.ndarray, List[float]]
            Optimized parameters and energy history
        """
        parameters = initial_parameters.copy()
        energy_history = []
        
        for iteration in range(n_iterations):
            # Compute current energy
            energy = self.manifold.energy_expectation(parameters)
            energy_history.append(energy)
            
            print(f"Iteration {iteration:3d}: Energy = {energy:.8f}")
            
            # Check convergence
            if iteration > 0 and abs(energy_history[-1] - energy_history[-2]) < tolerance:
                print(f"Converged at iteration {iteration}")
                break
            
            # Perform natural gradient step
            try:
                parameters = self.natural_gradient_step(parameters, learning_rate)
            except Exception as e:
                print(f"Optimization failed at iteration {iteration}: {e}")
                break
        
        return parameters, energy_history


class QuantumVariationalEigensolver:
    """
    Variational Quantum Eigensolver (VQE) with geodesic transport optimization
    
    Finds the ground state energy of a given Hamiltonian using
    variational quantum circuits and natural gradient descent.
    """
    
    def __init__(self, hamiltonian: np.ndarray, n_qubits: int, n_layers: int = 3):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize quantum circuit
        self.circuit = QuantumCircuit(n_qubits, n_layers)
        
        # Initialize manifold and geodesic transport
        self.manifold = QuantumParameterManifold(self.circuit, hamiltonian)
        self.geodesic_transport = GeodesicTransport(self.manifold)
        
        # Initialize optimizer
        self.optimizer = NaturalGradientOptimizer(self.manifold, self.geodesic_transport)
        
    def find_ground_state(self, n_iterations: int = 100, 
                         learning_rate: float = 0.01,
                         random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Find the ground state using VQE with geodesic optimization
        
        Returns:
        --------
        Dict[str, Any]
            Results containing optimized parameters, energy, and convergence history
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Initialize parameters randomly
        initial_parameters = np.random.uniform(-np.pi, np.pi, self.circuit.n_params)
        
        print("Starting VQE optimization with geodesic transport...")
        print(f"Hamiltonian size: {self.hamiltonian.shape}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Number of layers: {self.n_layers}")
        print(f"Number of parameters: {self.circuit.n_params}")
        print("-" * 50)
        
        # Run optimization
        optimal_parameters, energy_history = self.optimizer.optimize(
            initial_parameters, n_iterations, learning_rate
        )
        
        # Get final results
        final_energy = self.manifold.energy_expectation(optimal_parameters)
        final_state = self.circuit.state_vector(optimal_parameters)
        
        # Compute exact ground state for comparison (if small enough)
        exact_ground_energy = None
        if self.hamiltonian.shape[0] <= 16:  # Only for small systems
            eigenvalues = np.linalg.eigvals(self.hamiltonian)
            exact_ground_energy = np.min(eigenvalues)
        
        results = {
            'optimal_parameters': optimal_parameters,
            'final_energy': final_energy,
            'exact_ground_energy': exact_ground_energy,
            'energy_history': energy_history,
            'final_state': final_state,
            'n_iterations': len(energy_history),
            'converged': len(energy_history) < n_iterations
        }
        
        # Print summary
        print("-" * 50)
        print("VQE Optimization Complete!")
        print(f"Final energy: {final_energy:.8f}")
        if exact_ground_energy is not None:
            error = abs(final_energy - exact_ground_energy)
            print(f"Exact ground energy: {exact_ground_energy:.8f}")
            print(f"Error: {error:.8f}")
        print(f"Iterations: {len(energy_history)}")
        
        return results


def create_test_hamiltonian(n_qubits: int, hamiltonian_type: str = "ising") -> np.ndarray:
    """
    Create test Hamiltonians for VQE demonstrations
    
    Parameters:
    -----------
    n_qubits : int
        Number of qubits
    hamiltonian_type : str
        Type of Hamiltonian ("ising", "heisenberg", "random")
        
    Returns:
    --------
    np.ndarray
        Hamiltonian matrix
    """
    dim = 2 ** n_qubits
    
    if hamiltonian_type == "ising":
        # Transverse field Ising model: H = -J∑ZᵢZᵢ₊₁ - h∑Xᵢ
        H = np.zeros((dim, dim), dtype=complex)
        
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # ZZ interactions
        J = 1.0
        for i in range(n_qubits - 1):
            # Create ZᵢZᵢ₊₁ operator
            zz_op = np.eye(1, dtype=complex)
            for j in range(n_qubits):
                if j == i or j == i + 1:
                    zz_op = np.kron(zz_op, Z)
                else:
                    zz_op = np.kron(zz_op, I)
            H = H - J * zz_op
        
        # X field
        h = 0.5
        for i in range(n_qubits):
            x_op = np.eye(1, dtype=complex)
            for j in range(n_qubits):
                if j == i:
                    x_op = np.kron(x_op, X)
                else:
                    x_op = np.kron(x_op, I)
            H = H - h * x_op
            
    elif hamiltonian_type == "heisenberg":
        # Heisenberg model: H = J∑(XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁)
        H = np.zeros((dim, dim), dtype=complex)
        
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        J = 1.0
        for i in range(n_qubits - 1):
            for pauli in [X, Y, Z]:
                op = np.eye(1, dtype=complex)
                for j in range(n_qubits):
                    if j == i or j == i + 1:
                        op = np.kron(op, pauli)
                    else:
                        op = np.kron(op, I)
                H = H + J * op
                
    elif hamiltonian_type == "random":
        # Random Hermitian matrix
        H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = (H + H.conj().T) / 2  # Make Hermitian
        
    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")
    
    return H


def demonstrate_geodesic_vqe():
    """
    Demonstrate VQE with geodesic transport optimization
    """
    print("🌟 Variational Quantum Algorithms with Exact Geodesic Transport")
    print("=" * 70)
    
    # Test cases
    test_cases = [
        {"n_qubits": 2, "hamiltonian_type": "ising", "description": "2-qubit Ising model"},
        {"n_qubits": 3, "hamiltonian_type": "ising", "description": "3-qubit Ising model"},
        {"n_qubits": 2, "hamiltonian_type": "heisenberg", "description": "2-qubit Heisenberg model"}
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print("-" * 50)
        
        # Create Hamiltonian
        H = create_test_hamiltonian(case["n_qubits"], case["hamiltonian_type"])
        
        # Initialize VQE
        vqe = QuantumVariationalEigensolver(
            hamiltonian=H,
            n_qubits=case["n_qubits"],
            n_layers=3
        )
        
        # Run optimization
        result = vqe.find_ground_state(
            n_iterations=50,
            learning_rate=0.05,
            random_seed=42
        )
        
        results.append(result)
    
    # Plot results
    plot_optimization_results(results, test_cases)
    
    return results


def plot_optimization_results(results: List[Dict], test_cases: List[Dict]):
    """Plot optimization convergence for all test cases"""
    
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    for i, (result, case) in enumerate(zip(results, test_cases)):
        ax = axes[i]
        
        energy_history = result['energy_history']
        iterations = range(len(energy_history))
        
        # Plot energy convergence
        ax.plot(iterations, energy_history, 'b-', linewidth=2, label='VQE Energy')
        
        # Plot exact ground state if available
        if result['exact_ground_energy'] is not None:
            exact_energy = result['exact_ground_energy']
            ax.axhline(y=exact_energy, color='r', linestyle='--', 
                      linewidth=2, label=f'Exact: {exact_energy:.6f}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy')
        ax.set_title(f"{case['description']}\nFinal: {result['final_energy']:.6f}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle('VQE with Geodesic Transport - Convergence Results', 
                 fontsize=14, y=1.02)
    plt.show()


def analyze_quantum_geometry():
    """
    Analyze quantum geometric properties of parameter manifolds
    """
    print("\n🔍 Quantum Geometry Analysis")
    print("=" * 40)
    
    # Create a simple 2-qubit system
    n_qubits = 2
    H = create_test_hamiltonian(n_qubits, "ising")
    circuit = QuantumCircuit(n_qubits, 1)
    manifold = QuantumParameterManifold(circuit, H)
    
    # Sample points on the manifold
    n_samples = 20
    parameters_list = []
    qfim_eigenvalues = []
    energies = []
    
    print("Sampling quantum parameter manifold...")
    
    for i in range(n_samples):
        # Random parameters
        params = np.random.uniform(-np.pi, np.pi, circuit.n_params)
        parameters_list.append(params)
        
        # Compute QFIM eigenvalues (geometric properties)
        qfim = manifold.quantum_fisher_information_matrix(params)
        eigenvals = np.linalg.eigvals(qfim)
        qfim_eigenvalues.append(eigenvals)
        
        # Compute energy
        energy = manifold.energy_expectation(params)
        energies.append(energy)
    
    # Analyze results
    qfim_eigenvalues = np.array(qfim_eigenvalues)
    energies = np.array(energies)
    
    # Plot geometric analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: QFIM eigenvalue distribution
    axes[0].hist(qfim_eigenvalues.flatten(), bins=20, alpha=0.7, 
                 color='skyblue', density=True)
    axes[0].set_xlabel('QFIM Eigenvalues')
    axes[0].set_ylabel('Density')
    axes[0].set_title('QFIM Eigenvalue Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Condition number vs Energy
    condition_numbers = np.max(qfim_eigenvalues, axis=1) / (np.min(qfim_eigenvalues, axis=1) + 1e-10)
    axes[1].scatter(energies, condition_numbers, alpha=0.7, c='orange')
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('QFIM Condition Number')
    axes[1].set_title('Geometry vs Energy Landscape')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Geometric mean of eigenvalues
    geometric_means = np.exp(np.mean(np.log(qfim_eigenvalues + 1e-10), axis=1))
    axes[2].scatter(energies, geometric_means, alpha=0.7, c='green')
    axes[2].set_xlabel('Energy')
    axes[2].set_ylabel('QFIM Geometric Mean')
    axes[2].set_title('Manifold Volume vs Energy')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Quantum Parameter Manifold - Geometric Analysis', 
                 fontsize=14, y=1.02)
    plt.show()
    
    # Print statistics
    print(f"\nGeometric Statistics:")
    print(f"• Mean QFIM eigenvalue: {np.mean(qfim_eigenvalues):.6f}")
    print(f"• QFIM eigenvalue range: [{np.min(qfim_eigenvalues):.6f}, {np.max(qfim_eigenvalues):.6f}]")
    print(f"• Mean condition number: {np.mean(condition_numbers):.3f}")
    print(f"• Energy range: [{np.min(energies):.6f}, {np.max(energies):.6f}]")


def main():
    """
    Main demonstration of quantum geodesic transport VQA
    """
    print("🚀 Quantum Variational Algorithms with Exact Geodesic Transport")
    print("=" * 70)
    print("Implementation based on concepts from arXiv:2506.17395v2")
    print("Key features:")
    print("• Quantum Fisher Information Metric (QFIM)")
    print("• Geodesic transport on quantum parameter manifolds")
    print("• Natural gradient optimization")
    print("• Variational Quantum Eigensolver (VQE)")
    print("=" * 70)
    
    try:
        # Run VQE demonstrations
        vqe_results = demonstrate_geodesic_vqe()
        
        # Analyze quantum geometry
        analyze_quantum_geometry()
        
        print("\n✅ All demonstrations completed successfully!")
        
        return vqe_results
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    results = main()
