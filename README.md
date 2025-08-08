# VQE Geodesic Transport

Variational Quantum Eigensolver enhanced with quantum information geometry: natural (geodesic) gradient optimization using the Quantum Fisher Information Metric (QFIM), parallel transport, and exponential map updates. Implementations include a Qibo‑based simulator for realistic circuit execution and supporting experimental geometry utilities.

## Overview
This project explores geometry‑aware optimization for variational quantum algorithms, inspired by:

> Variational quantum algorithms with exact geodesic transport (arXiv:2506.17395v2)

The core idea is to treat the parameter space as a Riemannian manifold endowed with the QFIM, performing updates along geodesics rather than naive Euclidean steps.

## Key Features
- Qibo backend quantum circuit simulation
- Multiple ansätze (hardware efficient, alternating, strongly entangling)
- Quantum Fisher Information Matrix via parameter‑shift
- Natural gradient (QFIM inverse / pseudoinverse regularized)
- Discrete geodesic integration & exponential map updates
- Parallel transport of tangent vectors
- Adaptive learning rate strategy
- Interactive CLI for running single VQE, demos, or geometric analysis
- Visualization of convergence and final state probabilities

## Repository Structure
```
quantum_geodesic_transport/
  qibo_geodesic_vqa.py        # Qibo implementation with interactive main
  quantum_geodesic_vqa.py     # (Baseline / reference implementation)
  run_quantum_vqa.py          # Runner / helper script
  test_*.py                   # Tests / experiments
stiefel_manifolds/            # (Optional) Classical geometric visualization utilities
main.py                       # Entry for earlier geometry experiments
requirements.txt              # Python dependencies
```

## Getting Started
1. Create & activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (macOS/Linux)
   # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (If qibo not listed / desired backend tweaks):
   ```bash
   pip install qibo
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
