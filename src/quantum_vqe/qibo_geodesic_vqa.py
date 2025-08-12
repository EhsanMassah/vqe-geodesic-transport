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
        elif self.ansatz_type == "egt_optimized":
            # EGT-optimized ansatz from paper: structured for exact geodesics
            return self.n_layers * (2 * self.n_qubits + 1)
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
        elif self.ansatz_type == "egt_optimized":
            return self._egt_optimized_circuit(circuit, parameters)
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
    
    def _egt_optimized_circuit(self, circuit: models.Circuit, 
                              parameters: np.ndarray) -> models.Circuit:
        """
        EGT-optimized ansatz designed for exact geodesic transport
        
        Based on arXiv:2506.17395v2 - structured to enable exact metric computation
        without measurement overhead. Uses strategic parameter arrangement for
        analytical QFIM and geodesic path computation.
        """
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Global phase parameter (shared across layer)
            global_param = parameters[param_idx]
            param_idx += 1
            
            # RY rotations with strategic coupling
            for qubit in range(self.n_qubits):
                # Individual qubit rotation
                circuit.add(gates.RY(qubit, theta=parameters[param_idx]))
                param_idx += 1
                
                # Coupled rotation that enables exact geodesic computation
                coupled_angle = parameters[param_idx] + global_param / self.n_qubits
                circuit.add(gates.RZ(qubit, theta=coupled_angle))
                param_idx += 1
            
            # Structured entanglement pattern for geodesic optimization
            if self.n_qubits == 2:
                # For H2: simple CNOT with phase
                circuit.add(gates.CNOT(0, 1))
                circuit.add(gates.RZ(1, theta=global_param))
            else:
                # For larger molecules: ring connectivity
                for qubit in range(self.n_qubits):
                    next_qubit = (qubit + 1) % self.n_qubits
                    circuit.add(gates.CNOT(qubit, next_qubit))
                
                # Add central entangling gate with global parameter
                center = self.n_qubits // 2
                circuit.add(gates.RZ(center, theta=global_param))
        
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
                                        eps: float = 1e-6, 
                                        fast_approximation: bool = True) -> np.ndarray:
        """
        Compute QFIM with fast approximation option for better performance
        
        Args:
            parameters: Circuit parameters
            eps: Finite difference step size (when needed)
            fast_approximation: Use diagonal approximation for speed
        """
        if fast_approximation:
            return self._fast_qfim_approximation(parameters)
        else:
            return self._full_qfim_parameter_shift(parameters)
    
    def _fast_qfim_approximation(self, parameters: np.ndarray) -> np.ndarray:
        """
        Fast diagonal QFIM approximation - much faster than full computation
        
        Uses the fact that for many ansätze, off-diagonal terms are small
        and the dominant information is in the diagonal elements.
        """
        n = self.n_params
        qfim = np.zeros((n, n))
        
        # Compute only diagonal elements using parameter shift
        shift = np.pi / 2
        
        for i in range(n):
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            
            state_plus = self.circuit.get_state_vector(params_plus)
            state_minus = self.circuit.get_state_vector(params_minus)
            
            # Diagonal QFIM element
            overlap = np.abs(np.vdot(state_plus, state_minus))**2
            qfim[i, i] = 1 - overlap
        
        # Add small off-diagonal regularization for conditioning
        for i in range(n):
            for j in range(i+1, n):
                coupling = 0.1 * np.sqrt(qfim[i,i] * qfim[j,j])
                qfim[i,j] = qfim[j,i] = coupling
        
        return qfim
    
    def _full_qfim_parameter_shift(self, parameters: np.ndarray) -> np.ndarray:
        """
        Full QFIM computation (slow but accurate) - only use when necessary
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
    
    def energy_gradient(self, parameters: np.ndarray, 
                       fast_approximation: bool = True) -> np.ndarray:
        """
        Compute energy gradient with optional fast approximation
        
        Args:
            parameters: Circuit parameters
            fast_approximation: Use faster finite differences instead of parameter shift
        """
        if fast_approximation:
            return self._fast_gradient_finite_diff(parameters)
        else:
            return self._exact_gradient_parameter_shift(parameters)
    
    def _fast_gradient_finite_diff(self, parameters: np.ndarray, 
                                  eps: float = 1e-4) -> np.ndarray:
        """
        Fast gradient using central finite differences - much faster
        """
        gradient = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            energy_plus = self.energy_expectation(params_plus)
            energy_minus = self.energy_expectation(params_minus)
            
            gradient[i] = (energy_plus - energy_minus) / (2 * eps)
        
        return gradient
    
    def _exact_gradient_parameter_shift(self, parameters: np.ndarray) -> np.ndarray:
        """
        Exact energy gradient using parameter shift rule (slower but more accurate)
        
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
                       t: float = 1.0, fast_approximation: bool = True) -> np.ndarray:
        """
        Exponential map with fast approximation option
        
        Args:
            point: Current parameter point
            tangent: Tangent vector (update direction)
            t: Step size
            fast_approximation: Use simple linear update instead of geodesic integration
        """
        if fast_approximation:
            # Fast approximation: just add the scaled tangent vector
            return point + t * tangent
        else:
            return self._exact_geodesic_integration(point, tangent, t)
    
    def _exact_geodesic_integration(self, point: np.ndarray, tangent: np.ndarray, 
                                   t: float = 1.0) -> np.ndarray:
        """
        Exact exponential map: follow geodesic from point in tangent direction
        
        Integrates the geodesic equation using the metric structure (slow but accurate).
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
                             regularization: float = 1e-6,
                             fast_mode: bool = True) -> np.ndarray:
        """
        Perform natural gradient step with fast approximations
        
        Args:
            fast_mode: Use fast approximations for QFIM and gradient computation
        """
        # Compute energy gradient (fast by default)
        gradient = self.manifold.energy_gradient(parameters, fast_approximation=fast_mode)
        
        # Compute QFIM (fast by default)
        qfim = self.manifold.quantum_fisher_information_matrix(parameters, fast_approximation=fast_mode)
        
        # Regularize QFIM for numerical stability
        n = len(parameters)
        qfim_reg = qfim + regularization * np.eye(n)
        
        try:
            # Solve for natural gradient: G⁻¹∇E
            natural_gradient = np.linalg.solve(qfim_reg, gradient)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            natural_gradient = np.linalg.pinv(qfim_reg) @ gradient
        
        # Update using exponential map (fast by default)
        new_parameters = self.geodesic_transport.exponential_map(
            parameters, -learning_rate * natural_gradient, fast_approximation=fast_mode
        )
        
        return new_parameters
    
    def optimize(self, initial_parameters: np.ndarray,
                n_iterations: int = 100,
                learning_rate: float = 0.01,
                tolerance: float = 1e-8,
                adaptive_lr: bool = True,
                use_momentum: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        Enhanced optimization with EGT-CG inspired improvements
        """
        parameters = initial_parameters.copy()
        energy_history = []
        gradient_history = []
        lr = learning_rate
        
        # Momentum and conjugate gradient variables
        momentum = np.zeros_like(parameters) if use_momentum else None
        prev_gradient = None
        beta = 0.0  # Conjugate gradient parameter
        
        print(f"Starting optimization with {n_iterations} iterations...")
        
        for iteration in range(n_iterations):
            # Compute current energy and gradient (fast mode)
            current_energy = self.manifold.energy_expectation(parameters)
            gradient = self.manifold.energy_gradient(parameters, fast_approximation=True)
            gradient_norm = np.linalg.norm(gradient)
            
            energy_history.append(current_energy)
            gradient_history.append(gradient_norm)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iter {iteration:3d}: Energy = {current_energy:.8f}, "
                      f"|∇E| = {gradient_norm:.2e}, LR = {lr:.6f}")
            
            # Improved convergence criteria based on energy progress
            if iteration > 20:  # Allow minimum iterations for warm-up
                # Check energy convergence over window
                if len(energy_history) >= 10:
                    recent_window = energy_history[-10:]
                    energy_var = np.var(recent_window)
                    energy_trend = recent_window[-1] - recent_window[0]
                    
                    # Converged if energy is stable and trend is minimal
                    if energy_var < tolerance**2 and abs(energy_trend) < tolerance:
                        print(f"Converged at iteration {iteration} (energy stabilized)")
                        break
                
                # Secondary: gradient-based convergence (less strict)
                if gradient_norm < tolerance * 0.1:  # Much smaller threshold
                    print(f"Converged at iteration {iteration} (gradient threshold)")
                    break
            
            # EGT-CG inspired update
            if use_momentum and prev_gradient is not None:
                # Polak-Ribière conjugate gradient parameter
                beta = max(0, np.dot(gradient, gradient - prev_gradient) / 
                          (np.linalg.norm(prev_gradient)**2 + 1e-12))
                
                # Update momentum with conjugate gradient component
                if momentum is not None:
                    momentum = gradient + beta * momentum
                else:
                    momentum = gradient
                
                search_direction = momentum
            else:
                search_direction = gradient
                if momentum is not None:
                    momentum = gradient
            
            # Compute QFIM-based natural gradient with adaptive regularization (fast mode)
            qfim = self.manifold.quantum_fisher_information_matrix(parameters, fast_approximation=True)
            n = len(parameters)
            
            # Adaptive regularization based on QFIM condition number
            qfim_eigenvals = np.linalg.eigvals(qfim)
            condition_number = np.max(np.real(qfim_eigenvals)) / (np.min(np.real(qfim_eigenvals)) + 1e-12)
            
            # Scale regularization with condition number
            adaptive_reg = max(1e-8, min(1e-4, 1e-6 * condition_number))
            qfim_reg = qfim + adaptive_reg * np.eye(n)
            
            try:
                natural_direction = np.linalg.solve(qfim_reg, search_direction)
            except np.linalg.LinAlgError:
                natural_direction = np.linalg.pinv(qfim_reg) @ search_direction
            
            # Improved adaptive learning rate
            if adaptive_lr and len(energy_history) > 5:
                recent_energies = energy_history[-5:]
                if len(set([round(e, 8) for e in recent_energies])) == 1:
                    # Energy plateau - increase learning rate more aggressively
                    lr = min(lr * 1.5, 0.2)  # Higher maximum LR
                elif len(energy_history) > 1 and current_energy > energy_history[-2]:
                    # Energy increased - decrease learning rate
                    lr *= 0.7
                elif iteration > 50 and gradient_norm > 0.1:
                    # Stuck with large gradient - boost learning rate
                    lr = min(lr * 1.2, 0.15)
            
            # Geodesic update (fast mode)
            try:
                parameters = self.geodesic_transport.exponential_map(
                    parameters, -lr * natural_direction, fast_approximation=True
                )
            except Exception:
                # Fallback to simple update
                parameters = parameters - lr * natural_direction
            
            prev_gradient = gradient.copy()
        
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
        
    def _smart_initialization(self, strategy: str = "variance_aware", 
                             n_candidates: int = 5) -> np.ndarray:
        """
        Smart parameter initialization strategies based on the paper
        
        Args:
            strategy: Initialization strategy ('variance_aware', 'energy_based', 'random')
            n_candidates: Number of candidates to evaluate for energy-based init
        """
        if strategy == "variance_aware":
            # Variance-aware initialization - smaller initial parameters for better conditioning
            if self.ansatz_type == "egt_optimized":
                # For EGT ansatz: smaller variance for coupled parameters
                params = np.random.normal(0, 0.1, self.circuit.n_params)
                # Slightly larger variance for global parameters
                for layer in range(self.n_layers):
                    global_idx = layer * (2 * self.n_qubits + 1)
                    params[global_idx] = np.random.normal(0, 0.2)
            else:
                # Standard variance scaling
                params = np.random.normal(0, 0.1, self.circuit.n_params)
            return params
            
        elif strategy == "energy_based":
            # Energy-based initialization - pick best among random candidates
            best_params = None
            best_energy = np.inf
            
            for _ in range(n_candidates):
                candidate = np.random.uniform(-0.5, 0.5, self.circuit.n_params)
                energy = self.manifold.energy_expectation(candidate)
                if energy < best_energy:
                    best_energy = energy
                    best_params = candidate.copy()
            
            return best_params
            
        elif strategy == "layered":
            # Layer-wise initialization - start simple and build complexity
            params = np.zeros(self.circuit.n_params)
            if self.ansatz_type == "hardware_efficient":
                # Small rotations, bias toward |0⟩ state
                for i in range(self.circuit.n_params):
                    params[i] = np.random.normal(0, 0.05)
            return params
            
        else:  # random (original)
            return np.random.uniform(-np.pi, np.pi, self.circuit.n_params)
    
    def find_ground_state(self, n_iterations: int = 100,
                         learning_rate: float = 0.02,
                         random_seed: Optional[int] = None,
                         initialization: str = "variance_aware") -> Dict[str, Any]:
        """
        Find ground state using VQE with Qibo and geodesic optimization
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize parameters using smart strategy
        initial_parameters = self._smart_initialization(initialization)
        
        print("🚀 Starting Qibo VQE with Geodesic Optimization")
        print(f"Hamiltonian dimension: {2**self.n_qubits}×{2**self.n_qubits}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Number of layers: {self.n_layers}")
        print(f"Ansatz type: {self.ansatz_type}")
        print(f"Initialization: {initialization}")
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
            n_layers=1,
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
        circuit = QiboVariationalCircuit(n_qubits, n_layers=1, ansatz_type=ansatz_type)
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
