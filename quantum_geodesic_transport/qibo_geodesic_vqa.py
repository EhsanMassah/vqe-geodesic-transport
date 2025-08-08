#!/usr/bin/env python3
"""
Variational Quantum Algorithms with Exact Geodesic Transport - Qibo Implementation

Documentation reference: https://qibo.science/qibo/stable/getting-started/index.html

Implementation using Qibo quantum simulation framework for enhanced performance
and realistic quantum circuit simulation.

Based on arXiv:2506.17395v2 - "Variational quantum algorithms with exact geodesic transport"

Key features:
1. Qibo-based quantum circuit simulation
2. Hardware-efficient ansätze
3. Quantum Fisher Information Metric (QFIM) computation
4. Geodesic transport optimization
5. Natural gradient descent with parallel transport
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

try:
    import qibo
    from qibo import gates, models, hamiltonians, set_backend
    QIBO_AVAILABLE = True
except ImportError:
    print("⚠️  Qibo not installed. Installing Qibo...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qibo"])
    
    import qibo
    from qibo import gates, models, hamiltonians, set_backend
    QIBO_AVAILABLE = True

# Set Qibo backend
try:
    set_backend("numpy")
except:
    pass


class QiboVariationalCircuit:
    """
    Qibo-based variational quantum circuit for VQA
    
    Implements hardware-efficient ansätze with parameterized gates
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 2, ansatz_type: str = "hardware_efficient"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.n_params = self._count_parameters()
        self.backend = qibo.get_backend()
        
    def _count_parameters(self) -> int:
        """Count total number of parameters based on ansatz type"""
        if self.ansatz_type == "hardware_efficient":
            # Each layer: RY on each qubit + CNOT entanglers
            return self.n_qubits * self.n_layers
        elif self.ansatz_type == "alternating":
            # Each layer: RX, RZ on each qubit + entanglers
            return 2 * self.n_qubits * self.n_layers
        elif self.ansatz_type == "strongly_entangling":
            # Each layer: RX, RY, RZ on each qubit + full entanglement
            return 3 * self.n_qubits * self.n_layers
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
    
    def create_circuit(self, parameters: np.ndarray) -> models.Circuit:
        """
        Create parameterized quantum circuit using Qibo
        
        Parameters:
        -----------
        parameters : np.ndarray
            Circuit parameters
            
        Returns:
        --------
        qibo.models.Circuit
            Parameterized quantum circuit
        """
        circuit = models.Circuit(self.n_qubits)
        
        if self.ansatz_type == "hardware_efficient":
            return self._hardware_efficient_circuit(circuit, parameters)
        elif self.ansatz_type == "alternating":
            return self._alternating_circuit(circuit, parameters)
        elif self.ansatz_type == "strongly_entangling":
            return self._strongly_entangling_circuit(circuit, parameters)
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
    
    def _hardware_efficient_circuit(self, circuit: models.Circuit, 
                                   parameters: np.ndarray) -> models.Circuit:
        """Hardware-efficient ansatz with RY rotations and CNOT gates"""
        param_idx = 0
        
        for layer in range(self.n_layers):
            # RY rotations on all qubits
            for qubit in range(self.n_qubits):
                circuit.add(gates.RY(qubit, theta=parameters[param_idx]))
                param_idx += 1
            
            # CNOT entanglement pattern
            for qubit in range(self.n_qubits - 1):
                circuit.add(gates.CNOT(qubit, qubit + 1))
            
            # Optional: wrap-around CNOT for circular connectivity
            if self.n_qubits > 2:
                circuit.add(gates.CNOT(self.n_qubits - 1, 0))
        
        return circuit
    
    def _alternating_circuit(self, circuit: models.Circuit, 
                            parameters: np.ndarray) -> models.Circuit:
        """Alternating ansatz with RX and RZ rotations"""
        param_idx = 0
        
        for layer in range(self.n_layers):
            # RX rotations
            for qubit in range(self.n_qubits):
                circuit.add(gates.RX(qubit, theta=parameters[param_idx]))
                param_idx += 1
            
            # RZ rotations
            for qubit in range(self.n_qubits):
                circuit.add(gates.RZ(qubit, theta=parameters[param_idx]))
                param_idx += 1
            
            # Entanglement layer
            for qubit in range(0, self.n_qubits - 1, 2):
                circuit.add(gates.CNOT(qubit, qubit + 1))
            for qubit in range(1, self.n_qubits - 1, 2):
                circuit.add(gates.CNOT(qubit, qubit + 1))
        
        return circuit
    
    def _strongly_entangling_circuit(self, circuit: models.Circuit, 
                                    parameters: np.ndarray) -> models.Circuit:
        """Strongly entangling ansatz with full rotations"""
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Full rotation on each qubit
            for qubit in range(self.n_qubits):
                circuit.add(gates.RX(qubit, theta=parameters[param_idx]))
                param_idx += 1
                circuit.add(gates.RY(qubit, theta=parameters[param_idx]))
                param_idx += 1
                circuit.add(gates.RZ(qubit, theta=parameters[param_idx]))
                param_idx += 1
            
            # All-to-all entanglement (for small systems)
            if self.n_qubits <= 4:
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        circuit.add(gates.CNOT(i, j))
            else:
                # Linear entanglement for larger systems
                for qubit in range(self.n_qubits - 1):
                    circuit.add(gates.CNOT(qubit, qubit + 1))
        
        return circuit
    
    def get_state_vector(self, parameters: np.ndarray) -> np.ndarray:
        """Get the quantum state vector for given parameters"""
        circuit = self.create_circuit(parameters)
        result = self.backend.execute_circuit(circuit)
        return result.state()
    
    def get_density_matrix(self, parameters: np.ndarray) -> np.ndarray:
        """Get the density matrix for given parameters"""
        state_vector = self.get_state_vector(parameters)
        return np.outer(state_vector.conj(), state_vector)


class QiboQuantumManifold:
    """
    Quantum parameter manifold using Qibo backend
    
    Implements QFIM computation and energy calculations using
    realistic quantum circuit simulation.
    """
    
    def __init__(self, circuit: QiboVariationalCircuit, hamiltonian: hamiltonians.Hamiltonian):
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.n_params = circuit.n_params
        self.backend = qibo.get_backend()
        
    def quantum_fisher_information_matrix(self, parameters: np.ndarray, 
                                        eps: float = 1e-6) -> np.ndarray:
        """
        Compute QFIM using parameter shift rule for Qibo circuits
        
        For parameterized gates, uses the exact analytical formula:
        QFIM_ij = 4 * Re[⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩]
        """
        qfim = np.zeros((self.n_params, self.n_params), dtype=float)
        
        # Get current state
        state = self.circuit.get_state_vector(parameters)
        
        # Compute state derivatives using parameter shift
        state_derivatives = self._compute_state_derivatives_parameter_shift(parameters)
        
        for i in range(self.n_params):
            for j in range(self.n_params):
                # QFIM formula
                overlap_ij = np.vdot(state_derivatives[i], state_derivatives[j])
                overlap_i_state = np.vdot(state_derivatives[i], state)
                overlap_state_j = np.vdot(state, state_derivatives[j])
                
                qfim[i, j] = 4 * np.real(overlap_ij - overlap_i_state * overlap_state_j)
        
        return qfim
    
    def _compute_state_derivatives_parameter_shift(self, parameters: np.ndarray) -> List[np.ndarray]:
        """
        Compute state derivatives using parameter shift rule
        
        For rotation gates: ∂|ψ(θ)⟩/∂θ = (|ψ(θ+π/2)⟩ - |ψ(θ-π/2)⟩) / 2
        """
        derivatives = []
        shift = np.pi / 2
        
        for i in range(self.n_params):
            # Parameter shift rule
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            
            state_plus = self.circuit.get_state_vector(params_plus)
            state_minus = self.circuit.get_state_vector(params_minus)
            
            derivative = (state_plus - state_minus) / 2.0
            derivatives.append(derivative)
        
        return derivatives
    
    def energy_expectation(self, parameters: np.ndarray) -> float:
        """Compute energy expectation value using Qibo"""
        state = self.circuit.get_state_vector(parameters)
        
        # Convert Qibo Hamiltonian to matrix if needed
        if hasattr(self.hamiltonian, 'matrix'):
            H_matrix = self.hamiltonian.matrix
        else:
            H_matrix = self.hamiltonian
        
        energy = np.real(np.vdot(state, H_matrix @ state))
        return energy
    
    def energy_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute energy gradient using parameter shift rule
        
        ∂⟨H⟩/∂θ = (⟨H⟩(θ+π/2) - ⟨H⟩(θ-π/2)) / 2
        """
        gradient = np.zeros(self.n_params)
        shift = np.pi / 2
        
        for i in range(self.n_params):
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            
            energy_plus = self.energy_expectation(params_plus)
            energy_minus = self.energy_expectation(params_minus)
            
            gradient[i] = (energy_plus - energy_minus) / 2.0
        
        return gradient


class QiboGeodesicTransport:
    """
    Geodesic transport operations using Qibo quantum manifold
    
    Implements parallel transport and exponential map on the
    quantum parameter manifold with realistic circuit simulation.
    """
    
    def __init__(self, manifold: QiboQuantumManifold):
        self.manifold = manifold
        
    def parallel_transport_vector(self, vector: np.ndarray,
                                 start_point: np.ndarray,
                                 end_point: np.ndarray,
                                 n_steps: int = 20) -> np.ndarray:
        """
        Parallel transport a tangent vector along geodesic
        
        Uses discrete parallel transport with Christoffel symbols
        computed from the QFIM.
        """
        # Create path between points
        path = np.linspace(start_point, end_point, n_steps)
        transported_vector = vector.copy()
        
        for i in range(n_steps - 1):
            current_point = path[i]
            next_point = path[i + 1]
            tangent = next_point - current_point
            dt = 1.0 / (n_steps - 1)
            
            # Compute Christoffel symbols at current point
            christoffel = self._compute_christoffel_symbols(current_point)
            
            # Parallel transport equation: ∇_T V = 0
            # Discrete: V_new = V_old - dt * Γ(V, T)
            transport_correction = np.zeros_like(transported_vector)
            
            for j in range(len(transported_vector)):
                for k in range(len(tangent)):
                    for l in range(len(transported_vector)):
                        transport_correction[j] += (christoffel[j, k, l] * 
                                                   transported_vector[l] * 
                                                   tangent[k])
            
            transported_vector -= dt * transport_correction
            
        return transported_vector
    
    def _compute_christoffel_symbols(self, point: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols from QFIM
        
        Γᵢⱼₖ = ½ gⁱˡ (∂ⱼgₖₗ + ∂ₖgⱼₗ - ∂ₗgⱼₖ)
        """
        n = len(point)
        christoffel = np.zeros((n, n, n))
        eps = 1e-6
        
        # Get metric (QFIM) and its inverse
        metric = self.manifold.quantum_fisher_information_matrix(point)
        
        # Regularize for numerical stability
        metric += 1e-8 * np.eye(n)
        
        try:
            metric_inv = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            metric_inv = np.linalg.pinv(metric)
        
        # Compute metric derivatives numerically
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
        Exponential map: follow geodesic from point in tangent direction
        
        Integrates the geodesic equation using the metric structure.
        """
        n_steps = max(20, int(100 * t))
        dt = t / n_steps
        
        current_pos = point.copy()
        velocity = tangent.copy()
        
        # Integrate geodesic equation
        for step in range(n_steps):
            # Compute Christoffel symbols at current position
            christoffel = self._compute_christoffel_symbols(current_pos)
            
            # Geodesic acceleration: -Γ(v, v)
            acceleration = np.zeros_like(velocity)
            for i in range(len(velocity)):
                for j in range(len(velocity)):
                    for k in range(len(velocity)):
                        acceleration[i] -= christoffel[i, j, k] * velocity[j] * velocity[k]
            
            # Update position and velocity (Verlet integration)
            current_pos += velocity * dt + 0.5 * acceleration * dt**2
            velocity += acceleration * dt
        
        return current_pos


class QiboNaturalGradientOptimizer:
    """
    Natural gradient optimizer for Qibo-based VQA
    
    Implements optimization using the quantum Fisher information
    metric and geodesic transport operations.
    """
    
    def __init__(self, manifold: QiboQuantumManifold, 
                 geodesic_transport: QiboGeodesicTransport):
        self.manifold = manifold
        self.geodesic_transport = geodesic_transport
        
    def natural_gradient_step(self, parameters: np.ndarray, 
                             learning_rate: float = 0.01,
                             regularization: float = 1e-6) -> np.ndarray:
        """
        Perform natural gradient step with geodesic update
        
        Natural gradient: G⁻¹∇E where G is QFIM
        Update via exponential map for manifold-aware optimization
        """
        # Compute energy gradient
        gradient = self.manifold.energy_gradient(parameters)
        
        # Compute QFIM
        qfim = self.manifold.quantum_fisher_information_matrix(parameters)
        
        # Regularize QFIM for numerical stability
        n = len(parameters)
        qfim_reg = qfim + regularization * np.eye(n)
        
        try:
            # Solve for natural gradient: G⁻¹∇E
            natural_gradient = np.linalg.solve(qfim_reg, gradient)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            natural_gradient = np.linalg.pinv(qfim_reg) @ gradient
        
        # Update using exponential map (geodesic step)
        new_parameters = self.geodesic_transport.exponential_map(
            parameters, -learning_rate * natural_gradient
        )
        
        return new_parameters
    
    def optimize(self, initial_parameters: np.ndarray,
                n_iterations: int = 100,
                learning_rate: float = 0.01,
                tolerance: float = 1e-8,
                adaptive_lr: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        Run natural gradient optimization with adaptive learning rate
        """
        parameters = initial_parameters.copy()
        energy_history = []
        lr = learning_rate
        
        for iteration in range(n_iterations):
            # Compute current energy
            energy = self.manifold.energy_expectation(parameters)
            energy_history.append(energy)
            
            print(f"Iteration {iteration:3d}: Energy = {energy:.8f}, LR = {lr:.6f}")
            
            # Check convergence
            if iteration > 0:
                energy_diff = abs(energy_history[-1] - energy_history[-2])
                if energy_diff < tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
            
            try:
                # Perform natural gradient step
                new_parameters = self.natural_gradient_step(parameters, lr)
                
                # Adaptive learning rate
                if adaptive_lr and iteration > 0:
                    new_energy = self.manifold.energy_expectation(new_parameters)
                    if new_energy > energy:
                        lr *= 0.8  # Reduce learning rate if energy increased
                    elif energy_diff < 1e-6:
                        lr *= 1.1  # Increase learning rate if converging slowly
                
                parameters = new_parameters
                
            except Exception as e:
                print(f"Optimization failed at iteration {iteration}: {e}")
                break
        
        return parameters, energy_history


class QiboVQE:
    """
    Variational Quantum Eigensolver using Qibo with geodesic optimization
    
    Implements VQE for finding ground states using natural gradient
    descent on quantum parameter manifolds.
    """
    
    def __init__(self, hamiltonian: hamiltonians.Hamiltonian, 
                 n_qubits: int, n_layers: int = 2, 
                 ansatz_type: str = "hardware_efficient"):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        
        # Initialize Qibo circuit
        self.circuit = QiboVariationalCircuit(n_qubits, n_layers, ansatz_type)
        
        # Initialize manifold and geodesic transport
        self.manifold = QiboQuantumManifold(self.circuit, hamiltonian)
        self.geodesic_transport = QiboGeodesicTransport(self.manifold)
        
        # Initialize optimizer
        self.optimizer = QiboNaturalGradientOptimizer(self.manifold, self.geodesic_transport)
        
    def find_ground_state(self, n_iterations: int = 100,
                         learning_rate: float = 0.02,
                         random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Find ground state using VQE with Qibo and geodesic optimization
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize parameters
        initial_parameters = np.random.uniform(-np.pi, np.pi, self.circuit.n_params)
        
        print("🚀 Starting Qibo VQE with Geodesic Optimization")
        print(f"Hamiltonian dimension: {2**self.n_qubits}×{2**self.n_qubits}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Number of layers: {self.n_layers}")
        print(f"Ansatz type: {self.ansatz_type}")
        print(f"Number of parameters: {self.circuit.n_params}")
        print("-" * 60)
        
        # Run optimization
        optimal_parameters, energy_history = self.optimizer.optimize(
            initial_parameters, n_iterations, learning_rate
        )
        
        # Get final results
        final_energy = self.manifold.energy_expectation(optimal_parameters)
        final_state = self.circuit.get_state_vector(optimal_parameters)
        
        # Compute exact ground state for comparison
        exact_ground_energy = None
        if hasattr(self.hamiltonian, 'eigenvalues'):
            exact_eigenvals = self.hamiltonian.eigenvalues()
            exact_ground_energy = np.min(exact_eigenvals)
        elif hasattr(self.hamiltonian, 'matrix'):
            H_matrix = self.hamiltonian.matrix
            exact_eigenvals = np.linalg.eigvals(H_matrix)
            exact_ground_energy = np.min(np.real(exact_eigenvals))
        
        results = {
            'optimal_parameters': optimal_parameters,
            'final_energy': final_energy,
            'exact_ground_energy': exact_ground_energy,
            'energy_history': energy_history,
            'final_state': final_state,
            'circuit': self.circuit,
            'n_iterations': len(energy_history),
            'converged': len(energy_history) < n_iterations
        }
        
        # Print results
        print("-" * 60)
        print("🎯 Qibo VQE Optimization Complete!")
        print(f"Final energy: {final_energy:.8f}")
        if exact_ground_energy is not None:
            error = abs(final_energy - exact_ground_energy)
            print(f"Exact ground energy: {exact_ground_energy:.8f}")
            print(f"Absolute error: {error:.8f}")
            print(f"Relative error: {error/abs(exact_ground_energy)*100:.4f}%")
        print(f"Total iterations: {len(energy_history)}")
        
        return results


def create_qibo_hamiltonian(n_qubits: int, hamiltonian_type: str = "tfim", *, h: float = 1.0, delta: float = 1.0) -> hamiltonians.Hamiltonian:
    """Create Qibo Hamiltonians following official API.

    Supported types (aliases in parentheses):
      tfim : Transverse Field Ising Model ("ising") -> TFIM(nqubits, h)
      xxz  : Anisotropic Heisenberg ("heisenberg") -> XXZ(nqubits, delta)
      xxx  : Isotropic Heisenberg -> XXX(nqubits)
      xyz  : Fully anisotropic XYZ -> XYZ(nqubits)

    Args:
        n_qubits: Number of qubits.
        hamiltonian_type: Model identifier.
        h: Transverse field strength for TFIM.
        delta: Anisotropy parameter for XXZ.
    """
    t = hamiltonian_type.lower()
    # Alias normalization
    if t == "ising":
        t = "tfim"
    if t == "heisenberg":  # map generic name to XXZ by default
        t = "xxz"

    if t == "tfim":
        # Qibo signature: TFIM(nqubits, h=0.0, dense=True, backend=None)
        ham = hamiltonians.TFIM(n_qubits, h=h)
    elif t == "xxz":
        ham = hamiltonians.XXZ(n_qubits, delta=delta)
    elif t == "xxx":
        ham = hamiltonians.XXX(n_qubits)
    elif t == "xyz":
        ham = hamiltonians.XYZ(n_qubits)
    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")
    return ham


def demonstrate_qibo_vqe():
    """Demonstrate VQE using Qibo with geodesic optimization"""
    print("🌟 Qibo-based VQE with Geodesic Transport")
    print("=" * 50)
    test_cases = [
        {"n_qubits": 2, "hamiltonian_type": "tfim", "ansatz_type": "hardware_efficient", "description": "2-qubit TFIM (h=1.0)"},
        {"n_qubits": 3, "hamiltonian_type": "tfim", "ansatz_type": "hardware_efficient", "description": "3-qubit TFIM (h=1.0)"},
        {"n_qubits": 2, "hamiltonian_type": "xxz", "ansatz_type": "alternating", "description": "2-qubit XXZ (delta=1.0)"}
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print("-" * 50)
        
        # Create Qibo Hamiltonian
        hamiltonian = create_qibo_hamiltonian(case["n_qubits"], case["hamiltonian_type"])
        
        # Initialize VQE
        vqe = QiboVQE(
            hamiltonian=hamiltonian,
            n_qubits=case["n_qubits"],
            n_layers=2,
            ansatz_type=case["ansatz_type"]
        )
        
        # Run optimization
        result = vqe.find_ground_state(
            n_iterations=80,
            learning_rate=0.03,
            random_seed=42
        )
        
        result['case_info'] = case
        results.append(result)
    
    # Plot results
    plot_qibo_results(results)
    
    return results


def plot_qibo_results(results: List[Dict]):
    """Plot Qibo VQE optimization results"""
    
    n_cases = len(results)
    fig, axes = plt.subplots(2, n_cases, figsize=(5 * n_cases, 8))
    if n_cases == 1:
        axes = axes.reshape(-1, 1)
    
    for i, result in enumerate(results):
        case_info = result['case_info']
        
        # Plot 1: Energy convergence
        ax1 = axes[0, i]
        energy_history = result['energy_history']
        iterations = range(len(energy_history))
        
        ax1.plot(iterations, energy_history, 'b-', linewidth=2, label='VQE Energy')
        
        if result['exact_ground_energy'] is not None:
            exact_energy = result['exact_ground_energy']
            ax1.axhline(y=exact_energy, color='r', linestyle='--', 
                       linewidth=2, label=f'Exact: {exact_energy:.6f}')
            
            # Calculate and display error
            final_energy = result['final_energy']
            error = abs(final_energy - exact_energy)
            rel_error = error / abs(exact_energy) * 100
            ax1.text(0.05, 0.95, f'Error: {error:.6f}\nRel: {rel_error:.3f}%',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy')
        ax1.set_title(f"{case_info['description']}\nFinal: {result['final_energy']:.6f}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Circuit visualization or state analysis
        ax2 = axes[1, i]
        
        # Plot final state probabilities
        final_state = result['final_state']
        probabilities = np.abs(final_state)**2
        
        n_states = len(probabilities)
        state_labels = [f"|{i:0{case_info['n_qubits']}b}⟩" for i in range(n_states)]
        
        bars = ax2.bar(range(n_states), probabilities, alpha=0.7, color='green')
        ax2.set_xlabel('Quantum State')
        ax2.set_ylabel('Probability')
        ax2.set_title('Final State Probabilities')
        ax2.set_xticks(range(n_states))
        ax2.set_xticklabels(state_labels, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Highlight dominant states
        max_prob_idx = np.argmax(probabilities)
        bars[max_prob_idx].set_color('red')
    
    plt.tight_layout()
    plt.suptitle('Qibo VQE with Geodesic Transport - Results', 
                 fontsize=16, y=1.02)
    plt.show()


def analyze_qibo_circuit_properties():
    """
    Analyze properties of Qibo quantum circuits and manifolds
    """
    print("\n🔍 Qibo Quantum Circuit Analysis")
    print("=" * 40)
    
    # Create test system
    n_qubits = 3
    hamiltonian = create_qibo_hamiltonian(n_qubits, "tfim")
    
    # Test different ansätze
    ansatz_types = ["hardware_efficient", "alternating", "strongly_entangling"]
    
    fig, axes = plt.subplots(2, len(ansatz_types), figsize=(5 * len(ansatz_types), 8))
    
    for i, ansatz_type in enumerate(ansatz_types):
        print(f"\nAnalyzing {ansatz_type} ansatz...")
        
        # Create circuit and manifold
        circuit = QiboVariationalCircuit(n_qubits, n_layers=2, ansatz_type=ansatz_type)
        manifold = QiboQuantumManifold(circuit, hamiltonian)
        
        # Sample random parameters
        n_samples = 15
        energies = []
        qfim_traces = []
        qfim_determinants = []
        
        for _ in range(n_samples):
            params = np.random.uniform(-np.pi, np.pi, circuit.n_params)
            
            energy = manifold.energy_expectation(params)
            energies.append(energy)
            
            qfim = manifold.quantum_fisher_information_matrix(params)
            qfim_traces.append(np.trace(qfim))
            qfim_determinants.append(np.linalg.det(qfim + 1e-10 * np.eye(len(qfim))))
        
        # Plot energy landscape
        ax1 = axes[0, i]
        ax1.hist(energies, bins=8, alpha=0.7, color='blue', density=True)
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{ansatz_type}\nEnergy Landscape')
        ax1.grid(True, alpha=0.3)
        
        # Plot QFIM properties
        ax2 = axes[1, i]
        ax2.scatter(qfim_traces, qfim_determinants, alpha=0.7, c=energies, 
                   cmap='viridis', s=50)
        ax2.set_xlabel('QFIM Trace')
        ax2.set_ylabel('QFIM Determinant')
        ax2.set_title('Geometric Properties')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for energy
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Energy')
        
        print(f"  Parameters: {circuit.n_params}")
        print(f"  Energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}]")
        print(f"  QFIM trace range: [{np.min(qfim_traces):.4f}, {np.max(qfim_traces):.4f}]")
    
    plt.tight_layout()
    plt.suptitle('Qibo Circuit Analysis - Different Ansätze', fontsize=14, y=1.02)
    plt.show()


def _prompt_choice(prompt: str, options: List[str], default: Optional[str] = None) -> str:
    """Utility: prompt user to select from options."""
    while True:
        print(f"{prompt} ({'/'.join(options)})" + (f" [default: {default}]" if default else ""))
        choice = input('> ').strip().lower()
        if not choice and default:
            return default
        if choice in options:
            return choice
        print("Invalid choice. Try again.")


def _prompt_int(prompt: str, valid: List[int], default: Optional[int] = None) -> int:
    while True:
        print(f"{prompt} {valid}" + (f" [default: {default}]" if default is not None else ""))
        txt = input('> ').strip()
        if not txt and default is not None:
            return default
        if txt.isdigit():
            val = int(txt)
            if val in valid:
                return val
        print("Invalid integer. Try again.")


def run_interactive_session():
    """Interactive runner letting the user choose action and parameters."""
    print("\n🧭 Interactive Qibo Geodesic VQE Session")
    actions = {
        '1': 'Run single VQE',
        '2': 'Run full demo ( predefined cases )',
        '3': 'Geometry / ansatz analysis',
        '4': 'Quit'
    }
    while True:
        print("\nSelect action:")
        for k, v in actions.items():
            print(f"  {k}. {v}")
        act = input('> ').strip()
        if act not in actions:
            print("Invalid selection.")
            continue
        if act == '4':
            print("Exiting interactive session.")
            return None
        if act == '2':
            return {'mode': 'full_demo'}
        if act == '3':
            return {'mode': 'analysis'}
        # Single VQE
        # Supported ansatz types
        ansatz = _prompt_choice("Choose ansatz", ['hardware_efficient','alternating','strongly_entangling'], 'hardware_efficient')
        qubits = _prompt_int("Number of qubits", [2,3,4], 2)
        layers = _prompt_int("Number of layers", list(range(1,7)), 2)
        htype = _prompt_choice("Hamiltonian type", ['tfim','xxz','xxx','xyz'], 'tfim')
        iters = _prompt_int("Iterations", list(range(10, 501, 10)), 100)
        print("Learning rate (float) [default 0.03]:")
        lr_txt = input('> ').strip()
        try:
            lr = float(lr_txt) if lr_txt else 0.03
        except ValueError:
            lr = 0.03
        return {
            'mode': 'single', 'ansatz': ansatz, 'n_qubits': qubits,
            'layers': layers, 'hamiltonian': htype, 'iterations': iters,
            'learning_rate': lr
        }


def run_single_vqe(config: Dict[str, Any]):
    """Run a single VQE using user-provided configuration."""
    print("\n🔧 Running single VQE with configuration:")
    for k, v in config.items():
        if k != 'mode':
            print(f"  {k}: {v}")
    ham = create_qibo_hamiltonian(config['n_qubits'], config['hamiltonian'])
    vqe = QiboVQE(ham, config['n_qubits'], config['layers'], config['ansatz'])
    result = vqe.find_ground_state(
        n_iterations=config['iterations'],
        learning_rate=config['learning_rate'],
        random_seed=42
    )
    # Minimal plot for single run
    energies = result['energy_history']
    plt.figure(figsize=(5,3))
    plt.plot(range(len(energies)), energies, 'b-', lw=2)
    if result['exact_ground_energy'] is not None:
        plt.axhline(result['exact_ground_energy'], color='r', ls='--', lw=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Single VQE Convergence')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return result

# --- Modify main to include interactive option ---
# Preserve original main logic; insert interactive branch at start.
def main():
    """Main demonstration of Qibo-based geodesic VQE (interactive capable)."""
    print("🚀 Qibo Implementation: Variational Quantum Algorithms with Geodesic Transport")
    print("=" * 80)
    print("Features:")
    print("• Qibo quantum circuit simulation")
    print("• Hardware-efficient and alternating ansätze")
    print("• Quantum Fisher Information Metric computation")
    print("• Natural gradient descent with geodesic optimization")
    print("• Parallel transport on quantum parameter manifolds")
    print("=" * 80)

    # Interactive selection
    try:
        config = run_interactive_session()
    except Exception as _e:  # fallback non-interactive
        config = {'mode': 'full_demo'}

    try:
        print(f"Qibo backend: {qibo.get_backend()}")
        print(f"Qibo version: {qibo.__version__}")

        if config is None:
            return None
        if config['mode'] == 'full_demo':
            vqe_results = demonstrate_qibo_vqe()
            analyze_qibo_circuit_properties()
            print("\n✅ Completed full demo and analysis")
            return vqe_results
        elif config['mode'] == 'analysis':
            analyze_qibo_circuit_properties()
            print("\n✅ Analysis complete")
            return None
        elif config['mode'] == 'single':
            result = run_single_vqe(config)
            print("\n✅ Single VQE run complete")
            return result
        else:
            print("Unknown mode; exiting.")
            return None

    except Exception as e:
        print(f"\n❌ Error during Qibo demonstration: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()
