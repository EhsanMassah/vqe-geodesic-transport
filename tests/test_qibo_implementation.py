#!/usr/bin/env python3
"""
Test runner for Qibo-based Geodesic VQA

This script tests the Qibo implementation of variational quantum algorithms
with geodesic transport optimization.
"""

import sys
import subprocess
import numpy as np

def install_qibo():
    """Install Qibo if not available"""
    try:
        import qibo
        print(f"✅ Qibo already installed (version {qibo.__version__})")
        return True
    except ImportError:
        print("📦 Installing Qibo...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "qibo[default]"])
            import qibo
            print(f"✅ Qibo installed successfully (version {qibo.__version__})")
            return True
        except Exception as e:
            print(f"❌ Failed to install Qibo: {e}")
            return False

def test_qibo_basic_functionality():
    """Test basic Qibo functionality"""
    try:
        import qibo
        from qibo import gates, models, hamiltonians, set_backend
        
        print("🧪 Testing basic Qibo functionality...")
        
        # Set backend
        set_backend("numpy")
        
        # Create a simple circuit
        circuit = models.Circuit(2)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
        
        # Execute circuit
        result = circuit()
        state = result.state()
        
        # Check Bell state preparation
        expected_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        if np.allclose(np.abs(state), np.abs(expected_state)):
            print("✅ Basic circuit test passed")
        else:
            print("❌ Basic circuit test failed")
            return False
        
        # Test Hamiltonian
        tfim = hamiltonians.TFIM(2, h=1.0)
        eigenvals = tfim.eigenvalues()
        print(f"✅ TFIM eigenvalues: {eigenvals[:4]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic Qibo test failed: {e}")
        return False

def test_qibo_geodesic_vqa():
    """Test the Qibo geodesic VQA implementation"""
    try:
        from qibo_geodesic_vqa import (
            QiboVariationalCircuit,
            QiboQuantumManifold, 
            QiboVQE,
            create_qibo_hamiltonian,
            demonstrate_qibo_vqe
        )
        
        print("🧪 Testing Qibo Geodesic VQA implementation...")
        
        # Test circuit creation
        circuit = QiboVariationalCircuit(2, 2, "hardware_efficient")
        params = np.random.uniform(-np.pi, np.pi, circuit.n_params)
        state = circuit.get_state_vector(params)
        
        print(f"✅ Circuit test: {circuit.n_params} parameters, state norm = {np.linalg.norm(state):.6f}")
        
        # Test Hamiltonian creation
        hamiltonian = create_qibo_hamiltonian(2, "tfim")
        print(f"✅ Hamiltonian test: created TFIM for 2 qubits")
        
        # Test manifold
        manifold = QiboQuantumManifold(circuit, hamiltonian)
        energy = manifold.energy_expectation(params)
        gradient = manifold.energy_gradient(params)
        
        print(f"✅ Manifold test: energy = {energy:.6f}, gradient norm = {np.linalg.norm(gradient):.6f}")
        
        # Test small VQE
        print("🚀 Running small VQE test...")
        vqe = QiboVQE(hamiltonian, 2, 1, "hardware_efficient")
        result = vqe.find_ground_state(n_iterations=10, learning_rate=0.05, random_seed=42)
        
        print(f"✅ VQE test: final energy = {result['final_energy']:.6f}")
        
        if result['exact_ground_energy'] is not None:
            error = abs(result['final_energy'] - result['exact_ground_energy'])
            print(f"✅ VQE error: {error:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qibo Geodesic VQA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("🌟 Qibo Geodesic VQA Test Suite")
    print("=" * 50)
    
    # Install Qibo if needed
    if not install_qibo():
        print("❌ Cannot proceed without Qibo installation")
        return False
    
    # Test basic Qibo functionality
    if not test_qibo_basic_functionality():
        print("❌ Basic Qibo tests failed")
        return False
    
    # Test our implementation
    if not test_qibo_geodesic_vqa():
        print("❌ Geodesic VQA tests failed")
        return False
    
    print("\n✅ All tests passed! Running full demonstration...")
    
    # Run full demonstration
    try:
        from qibo_geodesic_vqa import main as demo_main
        results = demo_main()
        print("\n🎉 Full demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Full demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
