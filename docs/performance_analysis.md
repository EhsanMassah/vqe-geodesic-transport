## **Summary of Performance Issues and Additional Improvements Needed**

### **Current Status After Improvements:**
- ✅ **Implemented EGT-optimized ansatz** with 10 parameters (vs 4 for simple HE)
- ✅ **Added variance-aware initialization** (smaller parameter variance)
- ✅ **Enhanced optimization** with conjugate gradients and momentum
- ❌ **Still getting 12% error** and early convergence at iteration 2

### **Root Cause Analysis:**
1. **Too aggressive convergence criteria** - stopping at iteration 2
2. **Learning rate too small** for the improved ansatz complexity
3. **QFIM regularization** may be too strong, causing poor conditioning
4. **Need better line search** and curvature-aware updates

### **Critical Next Improvements (From Paper arXiv:2506.17395v2):**

#### **1. Fix Convergence Criteria**
- Current: Stops when gradient norm < tolerance (too strict)
- Paper approach: Monitor energy convergence over multiple iterations
- **Fix**: Change convergence to energy-based with proper windowing

#### **2. Implement Proper EGT-CG Algorithm**
- Current: Modified natural gradient with simple momentum
- Paper approach: Exact geodesic transport with conjugate gradients
- **Fix**: Implement true EGT-CG with proper search directions

#### **3. Better Learning Rate and Regularization**
- Current: LR=0.03, regularization=1e-6 
- Paper approach: Adaptive scaling based on curvature and search direction
- **Fix**: Curvature-aware learning rate, dynamic regularization

#### **4. Line Search on Manifold**
- Current: Fixed step size geodesic update
- Paper approach: Armijo line search along geodesic
- **Fix**: Implement geodesic line search for optimal step size

#### **5. Circuit Depth vs Optimization Trade-off**
- Current: 2-layer EGT circuit may still be underpowered for H2
- Paper insight: Balance expressivity vs optimization difficulty
- **Fix**: Try 3-4 layers for H2, adaptive layer selection

### **Implementation Priority:**
1. **IMMEDIATE**: Fix convergence criteria (energy-based)
2. **HIGH**: Implement proper EGT-CG search directions  
3. **MEDIUM**: Add manifold line search
4. **LOW**: Advanced regularization schemes

The paper shows 20x improvement is possible - we need these core algorithmic fixes!
