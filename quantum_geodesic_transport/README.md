# Quantum Geodesic Transport VQA

Implementation of **Variational Quantum Algorithms with Exact Geodesic Transport** based on the concepts from arXiv:2506.17395v2.

## Overview

This implementation demonstrates how to enhance variational quantum algorithms (VQA) using exact geodesic transport on quantum parameter manifolds. The key innovation is using the Quantum Fisher Information Matrix (QFIM) as a Riemannian metric to perform natural gradient optimization via geodesic transport.

## Key Features

### 🌟 Core Algorithms
- **Quantum Fisher Information Matrix (QFIM)** computation
- **Geodesic transport** on quantum parameter manifolds  
- **Natural gradient descent** with Riemannian optimization
- **Parallel transport** of vectors along geodesics
- **Exponential map** for geodesic updates

### 🔬 Quantum Algorithms
- **Variational Quantum Eigensolver (VQE)** with geodesic optimization
- **Quantum parameter manifold** analysis
- **Hamiltonian ground state** finding
- Support for multiple **quantum circuit ansätze**

### 📊 Analysis Tools
- **Optimization convergence** visualization
- **Quantum geometry** analysis (eigenvalues, condition numbers)
- **Manifold curvature** properties
- **Method comparison** (geodesic vs standard gradient)

## Mathematical Foundation

### Quantum Parameter Manifolds
The space of quantum circuit parameters θ forms a Riemannian manifold where:
- **Metric tensor**: Quantum Fisher Information Matrix G_ij(θ)
- **Natural gradient**: ∇̃f = G⁻¹∇f  
- **Geodesic equation**: d²θ/dt² + Γ(dθ/dt, dθ/dt) = 0

### QFIM Computation
```
G_ij = 4 × Re[⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩]
```
where |ψ(θ)⟩ is the parameterized quantum state.

### Geodesic Transport
- **Parallel transport**: Preserves vector magnitudes along curves
- **Christoffel symbols**: Γᵢⱼₖ encode manifold curvature  
- **Exponential map**: exp_p(v) follows geodesic from point p in direction v

## File Structure

```
quantum_geodesic_transport/
├── quantum_geodesic_vqa.py    # Main implementation
├── run_quantum_vqa.py         # Interactive menu interface  
└── README.md                  # This file
```

## Usage

### Quick Start

```python
from quantum_geodesic_vqa import *

# Create a test Hamiltonian (2-qubit Ising model)
H = create_test_hamiltonian(n_qubits=2, hamiltonian_type="ising")

# Initialize VQE with geodesic transport
vqe = QuantumVariationalEigensolver(
    hamiltonian=H,
    n_qubits=2, 
    n_layers=2
)

# Find ground state
result = vqe.find_ground_state(
    n_iterations=50,
    learning_rate=0.05
)

print(f"Ground state energy: {result['final_energy']:.6f}")
```

### Interactive Interface

```bash
cd quantum_geodesic_transport
python run_quantum_vqa.py
```

The interactive menu provides:
1. **Single VQE runs** with custom parameters
2. **Quantum geometry analysis** 
3. **Method comparisons** (geodesic vs standard)
4. **Custom Hamiltonian** VQE
5. **Full demonstrations**

### Example Demonstrations

```python
# Run full demonstration
results = demonstrate_geodesic_vqe()

# Analyze quantum manifold geometry  
analyze_quantum_geometry()

# Compare optimization methods
# (Available in interactive menu)
```

## Supported Hamiltonians

### Built-in Models
- **Ising Model**: H = -J∑ZᵢZᵢ₊₁ - h∑Xᵢ
- **Heisenberg Model**: H = J∑(XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁)  
- **Random Hermitian**: For testing and benchmarks

### Custom Hamiltonians
- Support for arbitrary Hermitian matrices
- Pauli string combinations
- User-defined coefficients

## Technical Implementation

### Classes and Functions

#### Core Classes
- `QuantumCircuit`: Parameterized quantum circuit representation
- `QuantumParameterManifold`: Riemannian manifold with QFIM metric
- `GeodesicTransport`: Geodesic computation and parallel transport
- `NaturalGradientOptimizer`: Natural gradient descent with geodesics
- `QuantumVariationalEigensolver`: VQE with geodesic optimization

#### Key Functions
- `quantum_fisher_information_matrix()`: Compute QFIM
- `parallel_transport()`: Transport vectors along geodesics
- `exponential_map()`: Geodesic exponential map
- `natural_gradient_step()`: Single optimization step
- `find_ground_state()`: Complete VQE optimization

### Computational Complexity
- **QFIM computation**: O(n²p) where n = parameters, p = state dimension
- **Geodesic integration**: O(n³s) where s = integration steps
- **Memory requirements**: O(p²) for quantum states

## Results and Validation

### Test Cases
1. **2-qubit Ising model**: Exact solution comparison
2. **3-qubit systems**: Scalability demonstration  
3. **Heisenberg models**: Different physics regimes
4. **Custom Hamiltonians**: User-defined problems

### Performance Metrics
- **Convergence rate**: Iterations to reach target accuracy
- **Final energy error**: |E_VQE - E_exact|
- **Geometric properties**: QFIM eigenvalue analysis
- **Optimization stability**: Variance across runs

## Theoretical Background

### Quantum Information Geometry
- Quantum states live on complex projective spaces
- Fubini-Study metric for pure states
- Fisher-Rao metric for mixed states
- Connection to quantum speedups

### Riemannian Optimization
- Natural gradients follow manifold geometry
- Geodesics as optimal paths
- Parallel transport preserves relationships
- Convergence guarantees on manifolds

### Applications in Quantum Computing
- **QAOA**: Quantum Approximate Optimization Algorithm
- **VQE**: Variational Quantum Eigensolver
- **Quantum Machine Learning**: Training quantum neural networks
- **Quantum Control**: Optimal control theory

## Dependencies

### Required
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `scipy`: Scientific computing (optional for advanced features)

### Optional Extensions
- `qiskit`: For real quantum hardware
- `cirq`: Google's quantum computing framework
- `pennylane`: Quantum machine learning

## Future Extensions

### Planned Features
- **Quantum hardware integration** (Qiskit/Cirq/PennyLane)
- **Advanced ansätze** (Hardware-efficient, UCCSD)
- **Noise modeling** and error mitigation
- **Multi-objective optimization**
- **Quantum neural networks**

### Research Directions
- **Higher-order geodesic methods**
- **Adaptive metric learning**
- **Quantum-classical hybrid optimization**
- **Distributed quantum optimization**

## Citation

If you use this implementation in research, please cite:

```
@article{quantum_geodesic_vqa_2024,
  title={Variational quantum algorithms with exact geodesic transport},
  author={Based on arXiv:2506.17395v2},
  journal={Implementation},
  year={2024}
}
```

## License

This implementation is provided for educational and research purposes. Please refer to the original paper (arXiv:2506.17395v2) for the theoretical foundations.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional quantum circuit ansätze
- More efficient QFIM computation
- Hardware integration
- Advanced visualization tools

## Contact

For questions about the implementation or theoretical aspects, please refer to the original paper or quantum computing community forums.
