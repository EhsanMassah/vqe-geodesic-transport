# VQE Geodesic Transport - Enhanced Performance Edition

Advanced Variational Quantum Eigensolver implementation featuring **200x performance improvements** through quantum information geometry optimization. Implements exact geodesic transport (EGT) with fast approximations for practical molecular electronic structure calculations.

## � Project Structure

```
├── src/                          # Core implementation
│   ├── quantum_vqe/             # Enhanced VQE with 200x speedup
│   │   ├── qibo_geodesic_vqa.py    # Fast geometric optimization engine
│   │   ├── qibo_molecular_vqe.py   # Interactive molecular VQE interface
│   │   └── quantum_geodesic_vqa.py # NumPy baseline implementation
│   └── geometry/                # Stiefel manifold visualization
│       ├── stiefel_manifold.py     # Core manifold operations
│       └── run_stiefel.py          # Interactive geometric demos
├── examples/                    # Ready-to-run demonstrations
│   ├── main.py                     # Main menu with quick options
│   ├── molecular_vqe.py           # Direct molecular VQE access
│   └── stiefel_visualization.py   # Geometric learning tools
├── tests/                       # Test suite
├── docs/                        # Comprehensive documentation
│   ├── quantum_vqe_guide.md       # VQE implementation details
│   ├── stiefel_geometry_guide.md  # Geometric foundations
│   ├── performance_analysis.md    # 200x speedup analysis
│   └── improvements_analysis.md   # Paper implementation notes
├── results/                     # Benchmark data and figures
└── requirements.txt             # Python dependencies
```

## �🚀 Latest Performance Breakthroughs (v2.0)

**Speed Improvements:**
- **200x faster execution**: 0.7s vs 130s for H2 optimization
- **Fast QFIM approximation**: Diagonal approximation replaces O(n²) computation
- **Optimized gradients**: Finite differences instead of parameter shifts
- **Smart geodesic updates**: Linear approximation with exact fallback

**Accuracy Improvements:**
- **3x better convergence**: 3.96% error vs 12.14% for H2 molecule
- **EGT-optimized ansatz**: Paper-inspired circuit design for exact geodesics
- **Variance-aware initialization**: Smart parameter initialization strategies
- **Adaptive learning rates**: Dynamic scaling based on optimization progress

## Overview
This project implements cutting-edge geometry‑aware optimization for variational quantum algorithms, based on:

> **Variational quantum algorithms with exact geodesic transport** (arXiv:2506.17395v2)

**Core Innovation:** Treats the quantum parameter space as a Riemannian manifold with the Quantum Fisher Information Metric (QFIM), enabling natural gradient updates along geodesics for superior convergence.

## 🎯 Key Features

### Performance & Speed
- **Fast approximation modes** with 200x speedup for practical use
- **Qibo backend** quantum circuit simulation with numpy optimization
- **Adaptive regularization** based on QFIM condition numbers
- **Smart convergence criteria** with energy-based stopping conditions

### Quantum Circuits & Ansätze
- **EGT-optimized ansatz**: Specially designed for exact geodesic computation
- **Hardware-efficient circuits**: Standard rotation + entanglement patterns
- **Alternating ansätze**: RX/RZ rotations with structured entanglement
- **Strongly entangling circuits**: Full rotation sets with all-to-all connectivity

### Geometric Optimization
- **Quantum Fisher Information Matrix** via fast diagonal approximation
- **Natural gradient descent** with QFIM⁻¹ updates
- **Geodesic transport** using exponential map and parallel transport
- **Christoffel symbols** computation for exact curvature (when needed)
- **Conjugate gradient momentum** for enhanced convergence

### Molecular Applications
- **Electronic structure VQE** for H2, LiH, BeH2 molecules
- **Multiple basis sets**: STO-3G variants with configurable bond lengths
- **Jordan-Wigner transformation** for fermion-to-qubit mapping
- **Interactive interface**: Simplified 4-question setup for easy use

## Repository Structure
```
## 🚀 Quick Start Guide

### Option 1: Interactive Main Menu (Recommended)
```bash
# Clone and setup
git clone https://github.com/EhsanMassah/vqe-geodesic-transport.git
cd vqe-geodesic-transport
python -m venv .venv
source .venv/bin/activate  # macOS/Linux (.venv\Scripts\activate on Windows)
pip install -r requirements.txt

# Run main menu
python examples/main.py
```

### Option 2: Direct Molecular VQE
```bash
# Enhanced molecular VQE with 200x speedup
python -m src.quantum_vqe.qibo_molecular_vqe

# Quick H2 test (recommended first run)
echo -e "1\n1\n4\nn\ny" | python -m src.quantum_vqe.qibo_molecular_vqe
```

### Option 3: Geometric Visualization
```bash
# Educational Stiefel manifold exploration
python -m src.geometry.run_stiefel
```

## Expected Performance (H2 Molecule)

| Method | Runtime | Error | Recommendation |
|--------|---------|-------|----------------|
| **EGT-Optimized (Option 4)** | **0.7s** | **3.96%** | ✅ **First choice** |
| Hardware Efficient (Option 1) | 2.1s | 5.2% | Good for learning |
| Alternating Ansatz (Option 2) | 1.8s | 4.8% | Balanced approach |
| Strongly Entangling (Option 3) | 3.2s | 4.1% | High accuracy needs |

## Installation & Dependencies

**System Requirements:**
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Qibo 0.2.20+ (quantum circuit simulation)

## Contributing & Research

**For Researchers:**
- All fast approximation methods include exact fallbacks for validation
- Extensive documentation of geometric algorithms and approximations
- Benchmarking tools for comparing different optimization approaches
- Open source implementation of cutting-edge quantum optimization

**For Developers:**
- Modular design allows easy extension to new ansätze
- Fast/exact mode toggles for development vs production
- Comprehensive test suite with performance regression monitoring
- Interactive tools for algorithm exploration and parameter tuning

## Citation

If you use this enhanced implementation in your research, please cite:

```bibtex
@article{ferreira2025variational,
  title={Variational quantum algorithms with exact geodesic transport},
  author={Ferreira-Martins, André J and others},
  journal={arXiv preprint arXiv:2506.17395v2},
  year={2025}
}
```

## License & Acknowledgments

This project implements and extends concepts from "Variational quantum algorithms with exact geodesic transport" (arXiv:2506.17395v2). The enhanced performance implementation includes novel fast approximation algorithms while preserving the theoretical foundations of geometric quantum optimization.

**Key Contributors:**
- Performance optimization algorithms and fast approximation methods
- Enhanced molecular VQE interface with smart defaults
- Comprehensive benchmarking and validation framework
- Educational geometric visualization tools

### 1. Core Geometric Engine
```python
from qibo_geodesic_vqa import QiboVariationalCircuit, QiboQuantumManifold

# Create EGT-optimized circuit
circuit = QiboVariationalCircuit(
    n_qubits=2, 
    n_layers=3, 
    ansatz_type="egt_optimized"
)

# Fast QFIM computation
manifold = QiboQuantumManifold(circuit, hamiltonian)
qfim = manifold.quantum_fisher_information_matrix(
    parameters, 
    fast_approximation=True  # 100x faster!
)
```

### 2. Molecular Electronic Structure
```python
from qibo_molecular_vqe import ElectronicStructureVQE

# Enhanced molecular VQE
vqe = ElectronicStructureVQE(
    molecule="H2",
    ansatz_type="egt_optimized",  # Paper-inspired ansatz
    n_layers=3,
    initial_params="variance_aware"  # Smart initialization
)

# Fast optimization
results = vqe.run_geodesic_optimization(
    iterations=50,
    learning_rate=0.08,  # Higher LR for improved ansatz
    fast_mode=True  # Enable all fast approximations
)
```

### 3. Interactive Analysis
```bash
python qibo_geodesic_vqa.py
# Choose: 3 (Geometry/ansatz analysis)
# Compares different ansätze performance
```

## Quick Usage
Run the interactive Qibo geodesic VQE session:
```bash
python quantum_geodesic_transport/qibo_geodesic_vqa.py
```
Select from: single run, full demo, or geometric / ansatz analysis.

Non‑interactive minimal example (inside a Python session):
```python
from quantum_geodesic_transport.qibo_geodesic_vqa import create_qibo_hamiltonian, QiboVQE
ham = create_qibo_hamiltonian(3, 'tfim')
vqe = QiboVQE(hamiltonian=ham, n_qubits=3, n_layers=2, ansatz_type='hardware_efficient')
result = vqe.find_ground_state(n_iterations=60, learning_rate=0.03, random_seed=42)
print(result['final_energy'], result['exact_ground_energy'])
```

## Geometric Methods Summary
- Metric: QFIM G(θ)
- Natural gradient direction: v = G^{-1} ∇E
- Geodesic update: θ_{new} ≈ Exp_{θ}( -η v ) via discrete integration using Christoffel symbols (finite differences of metric)
- Parallel transport: iterative correction using Γ^i_{jk}

Regularization (ε I) is applied when inverting / solving with the QFIM for numerical stability.

## Performance Notes
For larger qubit counts the naive finite difference computation of Christoffel symbols becomes expensive (O(P^4)). Future work: exploit structure, caching, or automatic differentiation when supported.

## Optional Classical Geometry Utilities
The `stiefel_manifolds/` module contains visualization tools for Stiefel manifolds used during early exploratory geometry experiments; they are not required for running the quantum VQE.

## Roadmap
- Caching & low‑rank updates of QFIM
- Alternative geodesic integrators (symplectic / adaptive)
- Backend extensions (GPU acceleration via qibojit if available)
- Benchmark vs standard gradient / Adam

## Citation
If you build upon the geometric VQE aspects, please cite the referenced arXiv work and acknowledge this repository.

## License
(Add a license here if needed.)

## Contributing
Issues & PRs welcome: focus on numerical stability, performance improvements, or additional ansätze.

---
Maintained by Ehsan Massah.
