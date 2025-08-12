# VQE Performance Improvements Based on arXiv:2506.17395v2

## Current Issues (H2 molecule, 12.14% error):
- Optimization stuck in local minima
- Constant gradient norm (~0.096) indicates poor convergence  
- Simple 1-layer ansatz may be insufficient
- Not implementing exact geodesic transport with conjugate gradients (EGT-CG)

## Key Improvements from Paper:

### 1. **Exact Geodesic Transport with Conjugate Gradients (EGT-CG)**
   - **Current**: Using simple natural gradient with Verlet integration
   - **Paper**: EGT-CG method with exact metric for analytical geodesic paths
   - **Benefit**: Up to 20x reduction in iterations vs standard methods
   - **Implementation**: Replace current geodesic integration with EGT-CG

### 2. **Improved Circuit Ansatz**
   - **Current**: Basic hardware-efficient with RY + CNOT
   - **Paper**: Specially designed ansatz for exact metric computation
   - **Benefit**: Enables analytical geodesic computation without measurement overhead
   - **Implementation**: Add paper-specific ansatz that allows exact QFIM computation

### 3. **Better Parameter Initialization**
   - **Current**: Random uniform initialization
   - **Paper**: Strategic initialization closer to optimal
   - **Benefit**: Avoids poor local minima from start
   - **Implementation**: Add multiple initialization strategies (ASGD, variance-aware)

### 4. **Adaptive Learning Rate with Momentum**
   - **Current**: Fixed learning rate with simple adaptive scaling
   - **Paper**: Sophisticated momentum and curvature-aware updates
   - **Benefit**: Better convergence in difficult landscapes
   - **Implementation**: Add momentum terms and adaptive scaling based on gradient history

### 5. **Degenerate Case Handling**
   - **Current**: No special handling for degenerate cases
   - **Paper**: Special methods for handling degenerate eigenspaces
   - **Benefit**: Rapid convergence even in challenging scenarios
   - **Implementation**: Add eigenvalue analysis and regularization

### 6. **Multi-Level Circuit Complexity**
   - **Current**: Fixed 1-layer (too simple for H2)
   - **Recommendation**: Progressive complexity increase
   - **Benefit**: Balance between expressivity and optimization difficulty
   - **Implementation**: Start with 2-3 layers for H2, scale appropriately

## Implementation Priority:
1. **Immediate**: Increase layers to 2-3 for H2 
2. **High**: Implement EGT-CG method
3. **Medium**: Add better ansatz and initialization
4. **Low**: Advanced momentum and degenerate handling
