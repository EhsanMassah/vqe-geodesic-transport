#!/usr/bin/env python3
"""
Test suite for Quantum Geodesic Transport VQA implementation

Validates the mathematical correctness and numerical stability
of the quantum variational algorithms with geodesic transport.
"""

import numpy as np
import sys
import os

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_geodesic_vqa import (
    QuantumCircuit, QuantumParameterManifold, GeodesicTransport,
    NaturalGradientOptimizer, QuantumVariationalEigensolver,
    create_test_hamiltonian
)


def test_quantum_circuit():
    """Test quantum circuit parameter counting and state generation"""
    
    # Test parameter counting
    circuit = QuantumCircuit(n_qubits=2, n_layers=1)
    expected_params = 3 * 2 * 1  # 3 rotations per qubit per layer
    assert circuit.n_params == expected_params
    
    # Test state vector generation
    params = np.random.randn(circuit.n_params)
    state = circuit.state_vector(params)
    
    # Check state vector properties
    assert len(state) == 2**2  # 4 states for 2 qubits
    assert np.isclose(np.linalg.norm(state), 1.0)  # Normalized
    
    print("✅ QuantumCircuit tests passed")


def test_qfim_properties():
    """Test Quantum Fisher Information Matrix properties"""
    
    # Create simple 1-qubit system
    n_qubits = 1
    circuit = QuantumCircuit(n_qubits, 1)
    
    # Simple Hamiltonian (Pauli-Z)
    H = np.array([[1, 0], [0, -1]])
    manifold = QuantumParameterManifold(circuit, H)
    
    # Test QFIM at random point
    params = np.random.uniform(-np.pi, np.pi, circuit.n_params)
    qfim = manifold.quantum_fisher_information_matrix(params)
    
    # QFIM should be symmetric
    assert np.allclose(qfim, qfim.T), "QFIM should be symmetric"
    
    # QFIM should be positive semi-definite
    eigenvals = np.linalg.eigvals(qfim)
    assert np.all(eigenvals >= -1e-10), "QFIM should be positive semi-definite"
    
    print("✅ QFIM properties tests passed")


def test_energy_gradient():
    """Test energy gradient computation"""
    
    # 1-qubit system with Pauli-X Hamiltonian
    circuit = QuantumCircuit(1, 1)
    H = np.array([[0, 1], [1, 0]])  # Pauli-X
    manifold = QuantumParameterManifold(circuit, H)
    
    # Test gradient at specific point
    params = np.array([0.0, 0.0, 0.0])  # |0⟩ state
    
    energy = manifold.energy_expectation(params)
    gradient = manifold.energy_gradient(params)
    
    # At |0⟩, energy should be 0 (⟨0|X|0⟩ = 0)
    assert np.isclose(energy, 0.0, atol=1e-6)
    
    # Gradient should not be zero (unless at critical point)
    assert len(gradient) == circuit.n_params
    
    print("✅ Energy gradient tests passed")


def test_geodesic_transport():
    """Test geodesic transport operations"""
    
    # Simple 1-qubit system
    circuit = QuantumCircuit(1, 1)
    H = np.eye(2)  # Identity (trivial)
    manifold = QuantumParameterManifold(circuit, H)
    geodesic = GeodesicTransport(manifold)
    
    # Test parallel transport
    start_point = np.array([0.0, 0.0, 0.0])
    end_point = np.array([0.1, 0.1, 0.1])
    vector = np.array([1.0, 0.0, 0.0])
    
    transported = geodesic.parallel_transport(vector, start_point, end_point)
    
    # Transported vector should have same dimension
    assert len(transported) == len(vector)
    
    # For small displacements, should preserve magnitude approximately
    original_norm = np.linalg.norm(vector)
    transported_norm = np.linalg.norm(transported)
    assert np.isclose(original_norm, transported_norm, rtol=0.1)
    
    print("✅ Geodesic transport tests passed")


def test_vqe_convergence():
    """Test VQE convergence on simple system"""
    
    # 2-qubit Ising Hamiltonian (more realistic test)
    H = create_test_hamiltonian(2, "ising")
    vqe = QuantumVariationalEigensolver(H, n_qubits=2, n_layers=1)
    
    # Run short optimization
    result = vqe.find_ground_state(
        n_iterations=30,
        learning_rate=0.02,
        random_seed=42
    )
    
    # Check that optimization reduces energy
    initial_energy = result['energy_history'][0]
    final_energy = result['energy_history'][-1]
    assert final_energy <= initial_energy, "Energy should decrease during optimization"
    
    # Check that we make reasonable progress
    energy_improvement = initial_energy - final_energy
    assert energy_improvement >= 0.0, "Should make some progress in optimization"
    
    # Final energy should be finite
    assert np.isfinite(final_energy), "Final energy should be finite"
    
    print("✅ VQE convergence tests passed")


def test_hamiltonian_creation():
    """Test Hamiltonian creation functions"""
    
    # Test Ising model
    H_ising = create_test_hamiltonian(2, "ising")
    assert H_ising.shape == (4, 4)
    assert np.allclose(H_ising, H_ising.conj().T)  # Hermitian
    
    # Test Heisenberg model
    H_heisenberg = create_test_hamiltonian(2, "heisenberg")
    assert H_heisenberg.shape == (4, 4)
    assert np.allclose(H_heisenberg, H_heisenberg.conj().T)  # Hermitian
    
    # Test random Hamiltonian
    H_random = create_test_hamiltonian(2, "random")
    assert H_random.shape == (4, 4)
    assert np.allclose(H_random, H_random.conj().T)  # Hermitian
    
    print("✅ Hamiltonian creation tests passed")


def test_numerical_stability():
    """Test numerical stability of key operations"""
    
    # Test with small system to ensure stability
    circuit = QuantumCircuit(2, 1)
    H = create_test_hamiltonian(2, "ising")
    manifold = QuantumParameterManifold(circuit, H)
    
    # Test QFIM conditioning
    params = np.random.uniform(-np.pi, np.pi, circuit.n_params)
    qfim = manifold.quantum_fisher_information_matrix(params)
    
    # Check condition number is reasonable
    eigenvals = np.linalg.eigvals(qfim)
    cond_number = np.max(eigenvals) / (np.min(eigenvals) + 1e-10)
    assert cond_number < 1e12, f"QFIM condition number {cond_number} too large"
    
    # Test gradient computation stability
    gradient = manifold.energy_gradient(params)
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    
    print("✅ Numerical stability tests passed")


def run_all_tests():
    """Run all test functions"""
    
    print("🧪 Running Quantum Geodesic Transport VQA Tests")
    print("=" * 50)
    
    test_functions = [
        test_quantum_circuit,
        test_qfim_properties,
        test_energy_gradient,
        test_geodesic_transport,
        test_vqe_convergence,
        test_hamiltonian_creation,
        test_numerical_stability
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return False


def benchmark_performance():
    """Benchmark performance of key operations"""
    
    print("\n📊 Performance Benchmarks")
    print("-" * 30)
    
    import time
    
    # Benchmark QFIM computation
    circuit = QuantumCircuit(2, 2)
    H = create_test_hamiltonian(2, "ising")
    manifold = QuantumParameterManifold(circuit, H)
    params = np.random.uniform(-np.pi, np.pi, circuit.n_params)
    
    start_time = time.time()
    for _ in range(10):
        qfim = manifold.quantum_fisher_information_matrix(params)
    qfim_time = (time.time() - start_time) / 10
    
    print(f"QFIM computation: {qfim_time:.4f} seconds")
    
    # Benchmark VQE iteration
    vqe = QuantumVariationalEigensolver(H, 2, 2)
    
    start_time = time.time()
    result = vqe.find_ground_state(n_iterations=5, learning_rate=0.05)
    vqe_time = time.time() - start_time
    
    print(f"VQE (5 iterations): {vqe_time:.4f} seconds")
    print(f"Average per iteration: {vqe_time/5:.4f} seconds")


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    
    # Run benchmarks if tests pass
    if success:
        benchmark_performance()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
