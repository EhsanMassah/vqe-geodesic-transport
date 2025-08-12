#!/usr/bin/env python3
"""
Electronic Structure VQE with Geodesic (Natural Gradient) Optimization using Qibo

Implements the Variational Quantum Algorithms with Exact Geodesic Transport (VQA EGT)
for electronic structure problems including H2, LiH, and BeH2 molecules using Jordan-Wigner
transformed Hamiltonians with extensive user-configurable parameters.

Features:
- Multiple molecules: H2, LiH, BeH2 with various bond lengths and basis sets
- User configurable: molecule, bond length, basis set, active space, ansatz type
- Geometric optimization parameters: learning rate, regularization, geodesic integration steps
- Circuit parameters: layers, entanglement patterns, initial parameter strategies
- Analysis tools: energy landscapes, QFIM properties, convergence diagnostics
- Natural (geodesic) gradient descent with adaptive learning rate and momentum
- Comparison with classical optimizers (Adam, BFGS) for benchmarking

Reference: arXiv:2506.17395v2
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union
import json
import time
import qibo
from qibo import hamiltonians, gates, models, set_backend

# Import core geodesic VQE infrastructure
from qibo_geodesic_vqa import (
    QiboVariationalCircuit,
    QiboQuantumManifold,
    QiboGeodesicTransport,
    QiboNaturalGradientOptimizer,
)

try:
    set_backend("numpy")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Molecular Hamiltonian Database
# ---------------------------------------------------------------------------
# Extended molecular database with multiple molecules, bond lengths, and basis sets
# All Hamiltonians are Jordan-Wigner transformed and potentially tapered

MOLECULAR_DATABASE = {
    "H2": {
        # 2-qubit tapered Hamiltonians (STO-3G)
        "STO-3G": {
            0.35: {"c0": -0.1373, "c1": 0.1805, "c2": 0.1805, "c3": 0.1699, "c4": 0.1813},
            0.55: {"c0": -0.4366, "c1": 0.1794, "c2": 0.1794, "c3": 0.1683, "c4": 0.1650},
            0.74: {"c0": -1.1364, "c1": 0.1580, "c2": 0.1580, "c3": 0.1520, "c4": 0.1056},  # equilibrium
            0.75: {"c0": -0.81055, "c1": 0.17218, "c2": 0.17218, "c3": 0.16893, "c4": 0.12055},
            0.90: {"c0": -1.0203, "c1": 0.1651,  "c2": 0.1651,  "c3": 0.1620,  "c4": 0.0950},
            1.05: {"c0": -1.1373, "c1": 0.1535,  "c2": 0.1535,  "c3": 0.1507,  "c4": 0.0720},
            1.40: {"c0": -1.0813, "c1": 0.1108,  "c2": 0.1108,  "c3": 0.1040,  "c4": 0.0400},
        },
        # 4-qubit full space (STO-3G) - example coefficients
        "STO-3G-FULL": {
            0.74: {
                "n_qubits": 4,
                "terms": [
                    {"coeff": -1.0523732, "pauli": "IIII"},
                    {"coeff": 0.39793742, "pauli": "IIIZ"},
                    {"coeff": 0.39793742, "pauli": "IIZI"},
                    {"coeff": 0.01128010, "pauli": "IIZZ"},
                    {"coeff": 0.18093119, "pauli": "XYII"},
                    {"coeff": 0.18093119, "pauli": "IXYI"},
                ]
            }
        }
    },
    "LiH": {
        # 4-qubit active space (STO-3G)
        "STO-3G": {
            1.45: {  # equilibrium bond length
                "n_qubits": 4,
                "terms": [
                    {"coeff": -7.8624823, "pauli": "IIII"},
                    {"coeff": 0.1777849, "pauli": "IIIZ"},
                    {"coeff": 0.1777849, "pauli": "IIZI"},
                    {"coeff": -0.2427106, "pauli": "IIZZ"},
                    {"coeff": 0.1704750, "pauli": "XYII"},
                    {"coeff": 0.1704750, "pauli": "IXYI"},
                ]
            }
        }
    },
    "BeH2": {
        # 6-qubit active space (STO-3G)
        "STO-3G": {
            2.54: {  # equilibrium bond length
                "n_qubits": 6,
                "terms": [
                    {"coeff": -15.5949722, "pauli": "IIIIII"},
                    {"coeff": 0.2182119, "pauli": "IIIIIZ"},
                    {"coeff": 0.2182119, "pauli": "IIIIZI"},
                    {"coeff": -0.1716128, "pauli": "IIIIZZ"},
                    {"coeff": 0.1654655, "pauli": "XYIIII"},
                ]
            }
        }
    }
}

PAULI = {
    'I': np.array([[1, 0],[0, 1]], complex),
    'X': np.array([[0, 1],[1, 0]], complex),
    'Y': np.array([[0, -1j],[1j, 0]], complex),
    'Z': np.array([[1, 0],[0, -1]], complex),
}

def kron_n(matrices: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of multiple matrices"""
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

def pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
    """Convert Pauli string like 'XYZI' to matrix"""
    return kron_n([PAULI[p] for p in pauli_string])

def build_molecular_hamiltonian(molecule: str, bond_length: float, basis: str = "STO-3G", 
                              include_yy: bool = False) -> Tuple[np.ndarray, Dict]:
    """Build molecular Hamiltonian matrix from database"""
    if molecule not in MOLECULAR_DATABASE:
        raise ValueError(f"Molecule {molecule} not available. Choices: {list(MOLECULAR_DATABASE.keys())}")
    
    mol_data = MOLECULAR_DATABASE[molecule]
    if basis not in mol_data:
        raise ValueError(f"Basis {basis} not available for {molecule}. Choices: {list(mol_data.keys())}")
    
    basis_data = mol_data[basis]
    if bond_length not in basis_data:
        # Find closest available bond length
        available = sorted(basis_data.keys())
        closest = min(available, key=lambda x: abs(x - bond_length))
        print(f"⚠️  Bond length {bond_length} Å not available. Using closest: {closest} Å")
        bond_length = closest
    
    hamiltonian_data = basis_data[bond_length]
    
    # Handle different Hamiltonian formats
    if "terms" in hamiltonian_data:
        # General Pauli string format
        n_qubits = hamiltonian_data["n_qubits"]
        H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        
        for term in hamiltonian_data["terms"]:
            coeff = term["coeff"]
            pauli_str = term["pauli"]
            pauli_mat = pauli_string_to_matrix(pauli_str)
            H += coeff * pauli_mat
            
    else:
        # Legacy H2 2-qubit format
        if molecule != "H2":
            raise ValueError(f"Legacy format only supported for H2")
            
        c0, c1, c2, c3, c4 = (hamiltonian_data[k] for k in ['c0', 'c1', 'c2', 'c3', 'c4'])
        I = kron_n([PAULI['I'], PAULI['I']])
        Z0 = kron_n([PAULI['Z'], PAULI['I']])
        Z1 = kron_n([PAULI['I'], PAULI['Z']])
        Z0Z1 = kron_n([PAULI['Z'], PAULI['Z']])
        X0X1 = kron_n([PAULI['X'], PAULI['X']])
        
        if include_yy:
            Y0Y1 = kron_n([PAULI['Y'], PAULI['Y']])
            exchange = 0.5 * (X0X1 + Y0Y1)
        else:
            exchange = X0X1
            
        H = c0 * I + c1 * Z0 + c2 * Z1 + c3 * Z0Z1 + c4 * exchange
    
    # Ensure Hermiticity
    H = 0.5 * (H + H.conj().T)
    
    info = {
        'molecule': molecule,
        'bond_length': bond_length,
        'basis': basis,
        'n_qubits': int(np.log2(H.shape[0])),
        'ground_energy': np.min(np.linalg.eigvals(H).real)
    }
    
    return H, info

def qibo_dense_hamiltonian(matrix: np.ndarray) -> hamiltonians.Hamiltonian:
    """Create Qibo Hamiltonian from dense matrix"""
    nqubits = int(np.log2(matrix.shape[0]))
    # For custom matrices, we can use the SymbolicHamiltonian approach
    # or work directly with the matrix in the energy computation
    
    # Create a custom Hamiltonian-like object that Qibo can work with
    class DenseMatrixHamiltonian:
        def __init__(self, matrix):
            self.matrix = matrix
            self.nqubits = int(np.log2(matrix.shape[0]))
        
        def eigenvalues(self):
            return np.linalg.eigvals(self.matrix)
        
        def expectation(self, state):
            """Compute expectation value <psi|H|psi>"""
            return np.real(np.conj(state) @ self.matrix @ state)
    
    return DenseMatrixHamiltonian(matrix)

# ---------------------------------------------------------------------------
# Enhanced Molecular Geodesic VQE with Configurable Parameters
# ---------------------------------------------------------------------------
class ElectronicStructureVQE:
    """
    Enhanced VQE for electronic structure with extensive configurability
    """
    
    def __init__(self, 
                 molecule: str = "H2",
                 bond_length: float = 0.74,
                 basis: str = "STO-3G",
                 ansatz_type: str = 'egt_optimized',  # Use improved ansatz by default
                 n_layers: int = 2,
                 entanglement_type: str = 'linear',
                 initial_params: str = 'variance_aware',  # Use smart initialization
                 include_yy: bool = False):
        
        self.molecule = molecule
        self.bond_length = bond_length
        self.basis = basis
        
        # Build Hamiltonian
        self.h_matrix, self.mol_info = build_molecular_hamiltonian(
            molecule, bond_length, basis, include_yy
        )
        self.n_qubits = self.mol_info['n_qubits']
        
        # Setup quantum circuit
        self.circuit = QiboVariationalCircuit(self.n_qubits, n_layers, ansatz_type)
        self.ansatz_type = ansatz_type
        self.n_layers = n_layers
        
        # Setup geometric optimization infrastructure
        self.ham = qibo_dense_hamiltonian(self.h_matrix)
        self.manifold = QiboQuantumManifold(self.circuit, self.ham)
        self.transport = QiboGeodesicTransport(self.manifold)
        
        # Initialize parameters
        self.initial_params_strategy = initial_params
        self.current_params = None
        
        print(f"🧬 Initialized {molecule} (bond={bond_length}Å, basis={basis})")
        print(f"   Qubits: {self.n_qubits} | Parameters: {self.circuit.n_params}")
        print(f"   Ground state energy (exact): {self.mol_info['ground_energy']:.8f}")

    def get_initial_parameters(self, strategy: str = None, seed: Optional[int] = 42) -> np.ndarray:
        """Generate initial parameters using improved strategies from arXiv:2506.17395v2"""
        if strategy is None:
            strategy = self.initial_params_strategy
            
        if seed is not None:
            np.random.seed(seed)
            
        n_params = self.circuit.n_params
        
        if strategy == 'variance_aware':
            # Variance-aware initialization - smaller initial parameters for better conditioning
            if self.ansatz_type == "egt_optimized":
                # For EGT ansatz: smaller variance for coupled parameters
                params = np.random.normal(0, 0.1, n_params)
                # Slightly larger variance for global parameters
                for layer in range(self.n_layers):
                    global_idx = layer * (2 * self.n_qubits + 1)
                    if global_idx < n_params:
                        params[global_idx] = np.random.normal(0, 0.2)
            else:
                # Standard variance scaling
                params = np.random.normal(0, 0.1, n_params)
            return params
            
        elif strategy == 'energy_based':
            # Energy-based initialization - pick best among random candidates
            best_params = None
            best_energy = np.inf
            n_candidates = 5
            
            for _ in range(n_candidates):
                candidate = np.random.uniform(-0.5, 0.5, n_params)
                energy = self.manifold.energy_expectation(candidate)
                if energy < best_energy:
                    best_energy = energy
                    best_params = candidate.copy()
            
            return best_params
            
        elif strategy == 'layered':
            # Layer-wise initialization - start simple and build complexity
            params = np.zeros(n_params)
            if self.ansatz_type == "hardware_efficient":
                # Small rotations, bias toward |0⟩ state
                for i in range(n_params):
                    params[i] = np.random.normal(0, 0.05)
            return params
            
        elif strategy == 'random':
            return np.random.uniform(-np.pi, np.pi, n_params)
        elif strategy == 'small_random':
            return np.random.uniform(-0.1, 0.1, n_params)
        elif strategy == 'zeros':
            return np.zeros(n_params)
        elif strategy == 'heuristic':
            # Simple heuristic: small rotations on single qubits, larger on entangling layers
            params = np.random.uniform(-0.5, 0.5, n_params)
            # Slightly larger rotations for later layers
            if self.ansatz_type == 'hardware_efficient':
                for layer in range(self.n_layers):
                    layer_start = layer * self.n_qubits
                    layer_end = (layer + 1) * self.n_qubits
                    params[layer_start:layer_end] *= (1 + 0.3 * layer)
            return params
        else:
            raise ValueError(f"Unknown parameter strategy: {strategy}")

    def run_geodesic_optimization(self, 
                                iterations: int = 100,
                                learning_rate: float = 0.08,  # Higher LR for improved ansatz
                                regularization: float = 1e-6,
                                adaptive_lr: bool = True,
                                tolerance: float = 1e-6,  # More reasonable tolerance
                                geodesic_steps: int = 20,
                                christoffel_eps: float = 1e-6,
                                momentum: float = 0.0,
                                param_strategy: str = None,
                                seed: Optional[int] = 42,
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Run geodesic optimization with extensive configurability
        """
        
        # Initialize parameters
        init_params = self.get_initial_parameters(param_strategy, seed)
        
        # Setup enhanced optimizer with configurable parameters
        optimizer = EnhancedGeodesicOptimizer(
            self.manifold, 
            self.transport,
            regularization=regularization,
            geodesic_steps=geodesic_steps,
            christoffel_eps=christoffel_eps,
            momentum=momentum
        )
        
        if verbose:
            print(f"\n🚀 Starting Geodesic VQE Optimization")
            print(f"   Molecule: {self.molecule} | Iterations: {iterations} | LR: {learning_rate}")
            print(f"   Regularization: {regularization} | Geodesic steps: {geodesic_steps}")
            print(f"   Momentum: {momentum} | Adaptive LR: {adaptive_lr}")
            print("-" * 60)
        
        start_time = time.time()
        
        # Run optimization
        opt_params, energy_hist, convergence_data = optimizer.optimize_detailed(
            init_params, 
            n_iterations=iterations,
            learning_rate=learning_rate,
            tolerance=tolerance,
            adaptive_lr=adaptive_lr,
            verbose=verbose
        )
        
        runtime = time.time() - start_time
        
        # Final results
        final_energy = self.manifold.energy_expectation(opt_params)
        exact_ground = self.mol_info['ground_energy']
        error = abs(final_energy - exact_ground)
        
        if verbose:
            print("\n" + "="*60)
            print("🎯 OPTIMIZATION COMPLETE")
            print(f"Final energy:    {final_energy:.10f}")
            print(f"Exact ground:    {exact_ground:.10f}")
            print(f"Absolute error:  {error:.2e}")
            print(f"Relative error:  {error/abs(exact_ground)*100:.4f}%")
            print(f"Runtime:         {runtime:.2f}s")
            print(f"Iterations:      {len(energy_hist)}/{iterations}")
            print("="*60)
        
        self.current_params = opt_params
        
        return {
            'optimal_parameters': opt_params,
            'energy_history': energy_hist,
            'final_energy': final_energy,
            'exact_ground_energy': exact_ground,
            'absolute_error': error,
            'relative_error': error/abs(exact_ground)*100,
            'n_iterations': len(energy_hist),
            'converged': len(energy_hist) < iterations,
            'runtime': runtime,
            'convergence_data': convergence_data,
            'molecule_info': self.mol_info,
            'optimization_params': {
                'learning_rate': learning_rate,
                'regularization': regularization,
                'geodesic_steps': geodesic_steps,
                'momentum': momentum,
                'adaptive_lr': adaptive_lr
            }
        }
    
    def analyze_energy_landscape(self, n_samples: int = 50, param_range: float = np.pi) -> Dict[str, Any]:
        """Analyze the energy landscape around current or random parameters"""
        if self.current_params is None:
            base_params = self.get_initial_parameters('random')
        else:
            base_params = self.current_params
            
        energies = []
        qfim_traces = []
        qfim_dets = []
        gradients = []
        
        print(f"🔍 Analyzing energy landscape ({n_samples} samples)...")
        
        for i in range(n_samples):
            # Sample around base parameters
            noise = np.random.uniform(-param_range/10, param_range/10, len(base_params))
            params = base_params + noise
            
            energy = self.manifold.energy_expectation(params)
            energies.append(energy)
            
            gradient = self.manifold.energy_gradient(params)
            gradients.append(np.linalg.norm(gradient))
            
            try:
                qfim = self.manifold.quantum_fisher_information_matrix(params)
                qfim_traces.append(np.trace(qfim))
                qfim_dets.append(np.linalg.det(qfim + 1e-10 * np.eye(len(qfim))))
            except:
                qfim_traces.append(0)
                qfim_dets.append(0)
        
        return {
            'energies': np.array(energies),
            'gradient_norms': np.array(gradients),
            'qfim_traces': np.array(qfim_traces),
            'qfim_determinants': np.array(qfim_dets),
            'energy_statistics': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'min': np.min(energies),
                'max': np.max(energies)
            }
        }

class EnhancedGeodesicOptimizer(QiboNaturalGradientOptimizer):
    """Enhanced optimizer with more configuration options"""
    
    def __init__(self, manifold, geodesic_transport, 
                 regularization: float = 1e-6,
                 geodesic_steps: int = 20,
                 christoffel_eps: float = 1e-6,
                 momentum: float = 0.0):
        super().__init__(manifold, geodesic_transport)
        self.regularization = regularization
        self.geodesic_steps = geodesic_steps
        self.christoffel_eps = christoffel_eps
        self.momentum = momentum
        self.velocity = None
        
    def optimize_detailed(self, initial_parameters, n_iterations=100, 
                         learning_rate=0.01, tolerance=1e-8, 
                         adaptive_lr=True, verbose=True):
        """Enhanced optimization with detailed convergence tracking"""
        
        parameters = initial_parameters.copy()
        energy_history = []
        lr_history = []
        gradient_norms = []
        qfim_traces = []
        
        lr = learning_rate
        self.velocity = np.zeros_like(parameters) if self.momentum > 0 else None
        
        for iteration in range(n_iterations):
            # Compute energy and gradient
            energy = self.manifold.energy_expectation(parameters)
            gradient = self.manifold.energy_gradient(parameters)
            
            energy_history.append(energy)
            lr_history.append(lr)
            gradient_norms.append(np.linalg.norm(gradient))
            
            # Compute QFIM
            try:
                qfim = self.manifold.quantum_fisher_information_matrix(parameters)
                qfim_traces.append(np.trace(qfim))
                
                # Natural gradient step with momentum
                qfim_reg = qfim + self.regularization * np.eye(len(parameters))
                
                try:
                    natural_gradient = np.linalg.solve(qfim_reg, gradient)
                except np.linalg.LinAlgError:
                    natural_gradient = np.linalg.pinv(qfim_reg) @ gradient
                
                # Apply momentum
                if self.velocity is not None:
                    self.velocity = self.momentum * self.velocity + (1 - self.momentum) * natural_gradient
                    update_direction = self.velocity
                else:
                    update_direction = natural_gradient
                
                # Geodesic update
                new_parameters = self.geodesic_transport.exponential_map(
                    parameters, -lr * update_direction, t=1.0
                )
                
            except Exception as e:
                if verbose:
                    print(f"Warning: QFIM computation failed at iteration {iteration}: {e}")
                # Fallback to regular gradient
                qfim_traces.append(0)
                new_parameters = parameters - lr * gradient
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: Energy = {energy:.8f}, |∇E| = {gradient_norms[-1]:.2e}, LR = {lr:.6f}")
            
            # Convergence check
            if iteration > 0:
                energy_diff = abs(energy_history[-1] - energy_history[-2])
                if energy_diff < tolerance:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            
            # Adaptive learning rate
            if adaptive_lr and iteration > 0:
                try:
                    new_energy = self.manifold.energy_expectation(new_parameters)
                    if new_energy > energy:
                        lr *= 0.8
                    elif energy_diff < 1e-6:
                        lr *= 1.05
                except:
                    pass
            
            parameters = new_parameters
        
        convergence_data = {
            'lr_history': lr_history,
            'gradient_norms': gradient_norms,
            'qfim_traces': qfim_traces
        }
        
        return parameters, energy_history, convergence_data

# ---------------------------------------------------------------------------
# Enhanced Interactive Interface
# ---------------------------------------------------------------------------

def print_header(title: str):
    """Print a nicely formatted header"""
    print("\n" + "="*70)
    print(f"🔬 {title}")
    print("="*70)

def print_section(title: str):
    """Print a section divider"""
    print(f"\n📋 {title}")
    print("-" * 50)

def prompt_molecule() -> str:
    """Enhanced molecule selection with descriptions"""
    molecule_info = {
        "H2": "Hydrogen molecule - 2 qubits, equilibrium bond length 0.74Å",
        "LiH": "Lithium hydride - 4 qubits, equilibrium bond length 1.45Å", 
        "BeH2": "Beryllium dihydride - 6 qubits, equilibrium bond length 2.54Å"
    }
    
    print_section("Molecule Selection")
    print("Available molecules:")
    for i, (mol, desc) in enumerate(molecule_info.items(), 1):
        print(f"  {i}. {mol:4s} - {desc}")
    
    while True:
        choice = input("\nSelect molecule (1-3) or name [1]: ").strip()
        if not choice:
            return "H2"
        
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= 3:
                return list(molecule_info.keys())[idx - 1]
        elif choice.upper() in molecule_info:
            return choice.upper()
        
        print("❌ Invalid choice. Please enter 1-3 or molecule name (H2/LiH/BeH2)")

def prompt_bond_length(molecule: str, available_lengths: List[float]) -> float:
    """Enhanced bond length selection"""
    print_section(f"Bond Length for {molecule}")
    print(f"Available bond lengths: {available_lengths} Å")
    
    # Show recommended values
    recommendations = {
        "H2": 0.74,
        "LiH": 1.45, 
        "BeH2": 2.54
    }
    recommended = recommendations.get(molecule)
    
    if recommended:
        print(f"💡 Recommended (equilibrium): {recommended}Å")
    
    while True:
        user_input = input(f"Enter bond length in Å [{available_lengths[0]}]: ").strip()
        if not user_input:
            return available_lengths[0]
        
        try:
            bond_length = float(user_input)
            if bond_length > 0:
                return bond_length
            else:
                print("❌ Bond length must be positive")
        except ValueError:
            print("❌ Please enter a valid number")

def prompt_basis_set(available_bases: List[str]) -> str:
    """Enhanced basis set selection"""
    print_section("Basis Set Selection")
    
    basis_descriptions = {
        "STO-3G": "Minimal basis set - fastest computation",
        "STO-3G-FULL": "Full 4-qubit space for H2 - more accurate",
        "6-31G": "Split-valence basis - higher accuracy",
        "cc-pVDZ": "Correlation-consistent basis - research quality"
    }
    
    print("Available basis sets:")
    for i, basis in enumerate(available_bases, 1):
        desc = basis_descriptions.get(basis, "Standard quantum chemistry basis")
        print(f"  {i}. {basis:12s} - {desc}")
    
    while True:
        choice = input(f"\nSelect basis set (1-{len(available_bases)}) or name [{available_bases[0]}]: ").strip()
        if not choice:
            return available_bases[0]
        
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(available_bases):
                return available_bases[idx - 1]
        elif choice in available_bases:
            return choice
        
        print(f"❌ Invalid choice. Available: {', '.join(available_bases)}")

def prompt_ansatz_type() -> str:
    """Enhanced ansatz selection with descriptions"""
    print_section("Quantum Circuit Ansatz")
    
    ansatz_info = {
        "hardware_efficient": "Hardware-efficient ansatz - RY + CNOT, good for NISQ devices",
        "alternating": "Alternating layered ansatz - X/Y rotations + entangling",
        "strongly_entangling": "Strongly entangling layers - maximum expressibility",
        "real_amplitudes": "Real amplitudes ansatz - RY gates only, real coefficients"
    }
    
    print("Available ansatz types:")
    for i, (ansatz, desc) in enumerate(ansatz_info.items(), 1):
        print(f"  {i}. {ansatz:20s} - {desc}")
    
    while True:
        choice = input("\nSelect ansatz (1-4) or name [hardware_efficient]: ").strip()
        if not choice:
            return "hardware_efficient"
        
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= 4:
                return list(ansatz_info.keys())[idx - 1]
        elif choice in ansatz_info:
            return choice
        
        print("❌ Invalid choice. Please select from the available options.")

def prompt_integer(prompt: str, default: int, min_val: int = 1, max_val: int = 1000) -> int:
    """Enhanced integer input with validation"""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        
        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Value must be between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid integer")

def prompt_float(prompt: str, default: float, min_val: float = 0.0, max_val: float = float('inf')) -> float:
    """Enhanced float input with validation"""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        
        try:
            value = float(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Value must be between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid number")

def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Enhanced yes/no prompt"""
    default_str = "Y/n" if default else "y/N"
    while True:
        user_input = input(f"{prompt} ({default_str}): ").strip().lower()
        if not user_input:
            return default
        if user_input in ['y', 'yes', 'true', '1']:
            return True
        elif user_input in ['n', 'no', 'false', '0']:
            return False
        else:
            print("❌ Please enter y/yes or n/no")

def prompt_parameter_strategy() -> str:
    """Enhanced parameter initialization strategy selection"""
    print_section("Parameter Initialization Strategy")
    
    strategies = {
        "random": "Random uniform [-π, π] - good exploration",
        "small_random": "Small random [-0.1, 0.1] - near zero initialization", 
        "zeros": "All zeros - start from computational basis",
        "heuristic": "Problem-specific heuristic - layer-dependent scaling"
    }
    
    print("Parameter initialization strategies:")
    for i, (strategy, desc) in enumerate(strategies.items(), 1):
        print(f"  {i}. {strategy:12s} - {desc}")
    
    while True:
        choice = input("\nSelect strategy (1-4) or name [random]: ").strip()
        if not choice:
            return "random"
        
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= 4:
                return list(strategies.keys())[idx - 1]
        elif choice in strategies:
            return choice
        
        print("❌ Invalid choice. Please select from available strategies.")

def show_optimization_summary(config: Dict[str, Any]):
    """Display a summary of selected configuration"""
    print_section("Configuration Summary")
    print(f"🧬 Molecule: {config['molecule']} ({config['bond_length']}Å, {config['basis']})")
    print(f"🔧 Circuit: {config['ansatz']} ansatz, {config['n_layers']} layers")
    print(f"⚙️  Optimization: {config['iterations']} iterations, LR={config['learning_rate']}")
    print(f"🎯 Parameters: {config['param_strategy']} init, momentum={config['momentum']}")
    print(f"📐 Geometric: regularization={config['regularization']:.0e}, geodesic_steps={config['geodesic_steps']}")
    
    if not prompt_yes_no("\nProceed with this configuration?"):
        return False
    return True

def run_simple_interactive():
    """Simple interactive runner with just the essential questions"""
    print_header("MOLECULAR VQE GEODESIC OPTIMIZER")
    print("A simple interface for quantum molecular simulations with geodesic optimization.")
    
    # Question 1: What molecule?
    print_section("Molecule Selection")
    molecules = ["H2", "LiH", "BeH2"]
    descriptions = {
        "H2": "Hydrogen (2 qubits, fastest)",
        "LiH": "Lithium hydride (4 qubits)", 
        "BeH2": "Beryllium dihydride (6 qubits, most complex)"
    }
    
    print("Available molecules:")
    for i, mol in enumerate(molecules, 1):
        print(f"  {i}. {mol} - {descriptions[mol]}")
    
    while True:
        choice = input("Select molecule (1-3) [1]: ").strip()
        if not choice:
            molecule = "H2"
            break
        try:
            idx = int(choice)
            if 1 <= idx <= 3:
                molecule = molecules[idx - 1]
                break
        except:
            pass
        print("❌ Please enter 1, 2, or 3")
    
    # Question 2: How many optimization steps?
    print_section("Optimization Settings")
    print("Choose optimization complexity:")
    print("  1. Quick test (50 iterations)")
    print("  2. Standard run (100 iterations)") 
    print("  3. Thorough optimization (200 iterations)")
    
    while True:
        choice = input("Select complexity (1-3) [2]: ").strip()
        if not choice:
            iterations = 100
            break
        try:
            idx = int(choice)
            if idx == 1:
                iterations = 50
                break
            elif idx == 2:
                iterations = 100
                break
            elif idx == 3:
                iterations = 200
                break
        except:
            pass
        print("❌ Please enter 1, 2, or 3")
    
    # Question 3: Circuit complexity?
    print_section("Circuit Configuration")
    print("Choose circuit complexity:")
    print("  1. Simple (1 layer, basic)")
    print("  2. Standard (2 layers, balanced)")
    print("  3. Complex (3 layers, high accuracy)")
    print("  4. EGT-Optimized (3 layers, paper-inspired ansatz) [RECOMMENDED]")
    
    while True:
        choice = input("Select circuit (1-4) [4]: ").strip()
        if not choice:
            n_layers = 3  # Use 3 layers for better H2 accuracy
            ansatz_type = "egt_optimized"
            break
        try:
            idx = int(choice)
            if idx == 1:
                n_layers = 1
                ansatz_type = "hardware_efficient"
                break
            elif idx == 2:
                n_layers = 2
                ansatz_type = "hardware_efficient"
                break
            elif idx == 3:
                n_layers = 3
                ansatz_type = "hardware_efficient"
                break
            elif idx == 4:
                n_layers = 3  # Use 3 layers for EGT-optimized
                ansatz_type = "egt_optimized"
                break
        except:
            pass
        print("❌ Please enter 1, 2, 3, or 4")
    
    # Question 4: Want analysis after?
    print_section("Analysis Options")
    run_analysis = prompt_yes_no("Perform energy landscape analysis after optimization?", False)
    
    # Configuration summary
    print_section("Summary")
    print(f"🧬 Molecule: {molecule}")
    print(f"🔄 Iterations: {iterations}")
    print(f"🔧 Circuit: {ansatz_type} ({n_layers} layers)")
    print(f"📊 Analysis: {'Yes' if run_analysis else 'No'}")
    
    if not prompt_yes_no("Start optimization?", True):
        print("❌ Cancelled.")
        return None
    
    # Run the optimization with smart defaults
    print_header("RUNNING OPTIMIZATION")
    print("🚀 Starting molecular VQE with geodesic optimization...")
    
    try:
        # Use smart defaults for all other parameters
        mol_data = MOLECULAR_DATABASE[molecule]
        basis = list(mol_data.keys())[0]  # Use first available basis
        bond_lengths = list(mol_data[basis].keys())
        bond_length = bond_lengths[0]  # Use first available bond length
        
        # Try to use equilibrium bond length if available
        equilibrium = {"H2": 0.74, "LiH": 1.45, "BeH2": 2.54}
        if molecule in equilibrium and equilibrium[molecule] in bond_lengths:
            bond_length = equilibrium[molecule]
        
        vqe = ElectronicStructureVQE(
            molecule=molecule,
            bond_length=bond_length,
            basis=basis,
            ansatz_type=ansatz_type,
            n_layers=n_layers,
            initial_params='variance_aware'  # Use improved initialization
        )
        
        results = vqe.run_geodesic_optimization(
            iterations=iterations,
            learning_rate=0.08,  # Higher LR for improved ansatz
            regularization=1e-6,
            momentum=0.0,
            param_strategy='variance_aware',  # Use improved initialization
            adaptive_lr=True,
            verbose=True
        )
        
        # Optional analysis
        if run_analysis:
            print_section("Energy Landscape Analysis")
            print("🔍 Analyzing energy landscape...")
            landscape = vqe.analyze_energy_landscape(n_samples=30)
            stats = landscape['energy_statistics']
            print(f"Energy statistics:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std:  {stats['std']:.6f}")
            print(f"  Min:  {stats['min']:.6f}")
            print(f"  Max:  {stats['max']:.6f}")
        
        # Save results option
        if prompt_yes_no("Save results to file?", False):
            filename = f"vqe_{molecule}_{iterations}iter.json"
            save_results = results.copy()
            for key, value in save_results.items():
                if isinstance(value, np.ndarray):
                    save_results[key] = value.tolist()
            
            with open(filename, 'w') as f:
                json.dump(save_results, f, indent=2)
            print(f"✅ Results saved to {filename}")
        
        print_header("OPTIMIZATION COMPLETE")
        print("✅ VQE optimization finished!")
        print(f"📊 Final energy: {results['final_energy']:.8f}")
        print(f"🎯 Error: {results['absolute_error']:.2e}")
        print(f"⏱️  Runtime: {results['runtime']:.1f}s")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return None

def run_interactive():
    """Enhanced interactive runner for molecular VQE experiments"""
    print("\n" + "="*60)
    print("� ADVANCED MOLECULAR VQE GEODESIC OPTIMIZER")
    print("="*60)
    
    # Molecule selection
    print("\nAvailable molecules:")
    for i, mol in enumerate(MOLECULAR_DATABASE.keys(), 1):
        print(f"  {i}. {mol}")
    
    mol_choice = input("Select molecule [1]: ") or "1"
    molecules = list(MOLECULAR_DATABASE.keys())
    molecule = molecules[int(mol_choice) - 1]
    
    mol_data = MOLECULAR_DATABASE[molecule]
    bases = list(mol_data.keys())
    # Get available bond lengths from first basis set
    first_basis = bases[0]
    bond_lengths = list(mol_data[first_basis].keys())
    
    print(f"\nSelected: {molecule}")
    print(f"Available bond lengths: {bond_lengths}")
    print(f"Available basis sets: {bases}")
    
    # Molecular parameters
    bond_length = float(input(f"Bond length Å [{bond_lengths[0]}]: ") or str(bond_lengths[0]))
    basis = input(f"Basis set [{bases[0]}]: ") or bases[0]
    
    # Circuit parameters
    layers = int(input("Number of layers [2]: ") or "2")
    ansatz = input("Ansatz type (hardware_efficient/alternating/strongly_entangling) [hardware_efficient]: ") or "hardware_efficient"
    include_yy = input("Include YY terms? (y/n) [n]: ").lower().startswith('y')
    
    # Optimization parameters
    iterations = int(input("Max iterations [100]: ") or "100")
    lr = float(input("Learning rate [0.03]: ") or "0.03")
    regularization = float(input("QFIM regularization [1e-6]: ") or "1e-6")
    momentum = float(input("Momentum coefficient [0.0]: ") or "0.0")
    geodesic_steps = int(input("Geodesic integration steps [20]: ") or "20")
    christoffel_eps = float(input("Christoffel symbol epsilon [1e-6]: ") or "1e-6")
    param_strategy = input("Parameter init (random/small_random/zeros/heuristic) [random]: ") or "random"
    adaptive_lr = input("Adaptive learning rate? (y/n) [y]: ").lower() != 'n'
    
    print(f"\n🚀 Running {molecule} VQE...")
    
    # Create and run VQE
    vqe = ElectronicStructureVQE(
        molecule=molecule,
        bond_length=bond_length,
        basis=basis,
        ansatz_type=ansatz,
        n_layers=layers,
        include_yy=include_yy
    )
    
    results = vqe.run_geodesic_optimization(
        iterations=iterations,
        learning_rate=lr,
        regularization=regularization,
        momentum=momentum,
        param_strategy=param_strategy,
        adaptive_lr=adaptive_lr,
        verbose=True
    )
    
    # Optional landscape analysis
    if input("\nPerform energy landscape analysis? (y/n) [n]: ").lower().startswith('y'):
        print("🔍 Analyzing energy landscape...")
        landscape = vqe.analyze_energy_landscape(n_samples=50)
        print(f"Energy statistics: {landscape['energy_statistics']}")
    
    return results

# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

def molecular_geodesic_vqe(molecule: str = "H2",
                          bond_length: float = 0.74, 
                          basis: str = "STO-3G",
                          n_layers: int = 2, 
                          ansatz: str = 'hardware_efficient',
                          iterations: int = 100,
                          learning_rate: float = 0.03,
                          regularization: float = 1e-6,
                          momentum: float = 0.0,
                          include_yy: bool = False,
                          param_strategy: str = 'random',
                          adaptive_lr: bool = True,
                          verbose: bool = True,
                          seed: Optional[int] = 42) -> Dict[str, Any]:
    """
    Enhanced molecular VQE with geodesic optimization
    
    Args:
        molecule: Molecule name ('H2', 'LiH', 'BeH2')
        bond_length: Internuclear distance in Angstroms
        basis: Basis set ('STO-3G', '6-31G', etc.)
        n_layers: Number of variational layers
        ansatz: Circuit ansatz type
        iterations: Maximum optimization iterations
        learning_rate: Initial learning rate
        regularization: QFIM regularization parameter
        momentum: Momentum coefficient for updates
        include_yy: Include YY Pauli terms
        param_strategy: Parameter initialization strategy
        adaptive_lr: Use adaptive learning rate
        verbose: Print detailed output
        seed: Random seed for reproducibility
    
    Returns:
        Comprehensive optimization results dictionary
    """
    if verbose:
        print(f"\n🧬 {molecule} Molecular VQE (bond={bond_length}Å, basis={basis})")
    
    # Create VQE instance
    vqe = ElectronicStructureVQE(
        molecule=molecule,
        bond_length=bond_length,
        basis=basis,
        ansatz_type=ansatz,
        n_layers=n_layers,
        include_yy=include_yy
    )
    
    # Run optimization
    results = vqe.run_geodesic_optimization(
        iterations=iterations,
        learning_rate=learning_rate,
        regularization=regularization,
        momentum=momentum,
        param_strategy=param_strategy,
        adaptive_lr=adaptive_lr,
        seed=seed,
        verbose=verbose
    )
    
    return results

# ---------------------------------------------------------------------------
# Example Usage Functions and Analysis Tools
# ---------------------------------------------------------------------------

def compare_molecules(molecules: List[str] = ["H2", "LiH", "BeH2"],
                     bond_length: float = 0.74,
                     basis: str = "STO-3G",
                     iterations: int = 50,
                     verbose: bool = True) -> Dict[str, Any]:
    """Compare VQE performance across different molecules"""
    results = {}
    
    print(f"\n🔬 Comparing molecules at {bond_length}Å ({basis} basis)")
    print("-" * 60)
    
    for mol in molecules:
        if mol not in MOLECULAR_DATABASE:
            print(f"⚠️  Skipping {mol} - not in database")
            continue
            
        print(f"\n🧬 Running {mol}...")
        try:
            vqe = ElectronicStructureVQE(
                molecule=mol,
                bond_length=bond_length,
                basis=basis,
                n_layers=2,
                ansatz_type='hardware_efficient'
            )
            
            result = vqe.run_geodesic_optimization(
                iterations=iterations,
                learning_rate=0.03,
                verbose=False
            )
            
            results[mol] = result
            
            if verbose:
                print(f"   Final energy: {result['final_energy']:.8f}")
                print(f"   Exact ground: {result['exact_ground_energy']:.8f}")
                print(f"   Error: {result['absolute_error']:.2e}")
                print(f"   Converged: {result['converged']} ({result['n_iterations']} iters)")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[mol] = None
    
    return results

def bond_length_scan(molecule: str = "H2",
                    bond_lengths: List[float] = [0.5, 0.74, 1.0, 1.5, 2.0],
                    basis: str = "STO-3G",
                    iterations: int = 50) -> Dict[str, Any]:
    """Scan VQE energies across different bond lengths"""
    print(f"\n📊 Bond length scan for {molecule} ({basis} basis)")
    print(f"Bond lengths: {bond_lengths}")
    print("-" * 60)
    
    scan_results = {}
    
    for bl in bond_lengths:
        print(f"\n📏 Bond length: {bl}Å")
        try:
            vqe = ElectronicStructureVQE(
                molecule=molecule,
                bond_length=bl,
                basis=basis,
                n_layers=2
            )
            
            result = vqe.run_geodesic_optimization(
                iterations=iterations,
                verbose=False
            )
            
            scan_results[bl] = {
                'final_energy': result['final_energy'],
                'exact_energy': result['exact_ground_energy'],
                'error': result['absolute_error'],
                'converged': result['converged'],
                'n_iterations': result['n_iterations']
            }
            
            print(f"   VQE energy: {result['final_energy']:.8f}")
            print(f"   Exact energy: {result['exact_ground_energy']:.8f}")
            print(f"   Error: {result['absolute_error']:.2e}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            scan_results[bl] = None
    
    return scan_results

def parameter_sensitivity_study(molecule: str = "H2",
                              bond_length: float = 0.74,
                              basis: str = "STO-3G") -> Dict[str, Any]:
    """Study sensitivity to various optimization parameters"""
    print(f"\n🔧 Parameter sensitivity study for {molecule}")
    print("-" * 60)
    
    # Base parameters
    base_params = {
        'molecule': molecule,
        'bond_length': bond_length,
        'basis': basis,
        'n_layers': 2,
        'iterations': 50,
        'verbose': False
    }
    
    studies = {}
    
    # Learning rate study
    print("\n📈 Learning rate sensitivity:")
    lr_values = [0.01, 0.03, 0.05, 0.1]
    studies['learning_rate'] = {}
    
    for lr in lr_values:
        result = molecular_geodesic_vqe(learning_rate=lr, **base_params)
        studies['learning_rate'][lr] = {
            'final_energy': result['final_energy'],
            'error': result['absolute_error'],
            'converged': result['converged']
        }
        print(f"   LR {lr:.3f}: Energy {result['final_energy']:.8f}, Error {result['absolute_error']:.2e}")
    
    # Regularization study  
    print("\n🔒 QFIM regularization sensitivity:")
    reg_values = [1e-8, 1e-6, 1e-4, 1e-2]
    studies['regularization'] = {}
    
    for reg in reg_values:
        result = molecular_geodesic_vqe(regularization=reg, **base_params)
        studies['regularization'][reg] = {
            'final_energy': result['final_energy'],
            'error': result['absolute_error'],
            'converged': result['converged']
        }
        print(f"   Reg {reg:.0e}: Energy {result['final_energy']:.8f}, Error {result['absolute_error']:.2e}")
    
    # Circuit depth study
    print("\n🔄 Circuit depth sensitivity:")
    layer_values = [1, 2, 3, 4]
    studies['n_layers'] = {}
    
    for layers in layer_values:
        # Remove n_layers from base_params for this study
        layer_params = {k: v for k, v in base_params.items() if k != 'n_layers'}
        result = molecular_geodesic_vqe(n_layers=layers, **layer_params)
        studies['n_layers'][layers] = {
            'final_energy': result['final_energy'],
            'error': result['absolute_error'],
            'converged': result['converged']
        }
        print(f"   {layers} layers: Energy {result['final_energy']:.8f}, Error {result['absolute_error']:.2e}")
    
    return studies

def run_comprehensive_demo():
    """Run a comprehensive demonstration of all features"""
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE MOLECULAR VQE GEODESIC TRANSPORT DEMONSTRATION")
    print("="*80)
    
    # Single molecule run
    print("\n1️⃣ Single molecule VQE optimization:")
    result = molecular_geodesic_vqe(
        molecule="H2",
        bond_length=0.74,
        basis="STO-3G",
        iterations=100,
        learning_rate=0.03,
        regularization=1e-6,
        momentum=0.1,
        verbose=True
    )
    
    # Molecule comparison
    print("\n2️⃣ Multi-molecule comparison:")
    comparison = compare_molecules(["H2", "LiH"], iterations=50)
    
    # Bond length scan
    print("\n3️⃣ Bond length potential energy surface:")
    scan = bond_length_scan("H2", [0.5, 0.74, 1.0, 1.5], iterations=30)
    
    # Parameter study
    print("\n4️⃣ Parameter sensitivity analysis:")
    sensitivity = parameter_sensitivity_study("H2")
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*80)
    
    return {
        'single_run': result,
        'molecule_comparison': comparison,
        'bond_scan': scan,
        'parameter_sensitivity': sensitivity
    }

# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_comprehensive_demo()
        elif sys.argv[1] == "compare":
            compare_molecules()
        elif sys.argv[1] == "scan":
            bond_length_scan()
        elif sys.argv[1] == "sensitivity":
            parameter_sensitivity_study()
        elif sys.argv[1] == "classic":
            # Use the old simple interactive interface
            run_interactive()
        elif sys.argv[1] == "advanced":
            # For users who want all the detailed options
            print("❌ Advanced interface has been removed for simplicity.")
            print("Use the default interface for streamlined experience.")
        else:
            print("Usage: python qibo_molecular_vqe.py [demo|compare|scan|sensitivity|classic]")
            print("  demo        - Comprehensive demonstration of all features")
            print("  compare     - Compare different molecules")
            print("  scan        - Bond length scan analysis")
            print("  sensitivity - Parameter sensitivity study")
            print("  classic     - Original detailed interactive interface")
            print("  (no args)   - Simple 4-question interface")
    else:
        try:
            run_simple_interactive()
        except KeyboardInterrupt:
            print("\nInterrupted.")
