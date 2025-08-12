"""
Quantum VQE Module

Enhanced quantum variational eigensolver with geodesic transport optimization.
Includes fast approximation algorithms and EGT-optimized ansatz designs.

Main Components:
- qibo_geodesic_vqa: Core geometric optimization engine
- qibo_molecular_vqe: Enhanced molecular VQE interface
- quantum_geodesic_vqa: NumPy baseline implementation
"""

from .qibo_geodesic_vqa import GeometricVQA, QuantumGeometry
from .qibo_molecular_vqe import ElectronicStructureVQE

__all__ = ['GeometricVQA', 'QuantumGeometry', 'ElectronicStructureVQE']
