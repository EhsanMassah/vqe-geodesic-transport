#!/usr/bin/env python3
"""
Stiefel Manifold Visualization

Visualizes the Stiefel manifold V(n,k) - the space of orthonormal k-frames in R^n.
For R³: V(3,1) - unit sphere S², V(3,2) - orthonormal 2-frames
For R⁴: V(4,1) - unit sphere S³, V(4,2) - orthonormal 2-frames in R⁴
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def generate_random_orthonormal_frame(n: int, k: int, num_samples: int = 1000) -> np.ndarray:
    """
    Generate random orthonormal k-frames in R^n using QR decomposition.
    
    Parameters:
    -----------
    n : int
        Dimension of ambient space
    k : int  
        Number of orthonormal vectors in each frame
    num_samples : int
        Number of random frames to generate
        
    Returns:
    --------
    np.ndarray
        Array of shape (num_samples, n, k) containing orthonormal frames
    """
    frames = np.zeros((num_samples, n, k))
    
    for i in range(num_samples):
        # Generate random matrix and apply QR decomposition
        random_matrix = np.random.randn(n, k)
        Q, R = qr(random_matrix)
        
        # Take only the first k columns of Q
        Q_k = Q[:, :k]
        
        # Ensure consistent orientation by making diagonal of R positive
        signs = np.sign(np.diag(R))
        if k == 1:
            Q_k = Q_k * signs[0]
        else:
            Q_k = Q_k @ np.diag(signs)
        
        frames[i] = Q_k
    
    return frames


def visualize_stiefel_v31():
    """
    Visualize V(3,1) - unit sphere S² in R³
    """
    # Generate points on unit sphere
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 25)
    PHI, THETA = np.meshgrid(phi, theta)
    
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere surface
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
    
    # Add some random points on the sphere
    frames = generate_random_orthonormal_frame(3, 1, 20)
    points = frames[:, :, 0]  # Extract the single vectors
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='red', s=50, alpha=0.8, label='Random points on V(3,1)')
    
    # Add coordinate axes
    ax.quiver([0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,1,0], [0,0,1], 
              color=['red', 'green', 'blue'], alpha=0.6, arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Stiefel Manifold V(3,1) = S² (Unit Sphere in R³)')
    ax.legend()
    
    plt.tight_layout()
    return fig


def visualize_stiefel_v32():
    """
    Visualize V(3,2) - orthonormal 2-frames in R³
    Each frame consists of two orthonormal vectors in R³
    """
    fig = plt.figure(figsize=(12, 5))
    
    # Generate random orthonormal 2-frames
    frames = generate_random_orthonormal_frame(3, 2, 15)
    
    # Plot 1: Show frames as vector pairs from origin
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(frames)))
    
    for i, frame in enumerate(frames[:8]):  # Show subset for clarity
        v1, v2 = frame[:, 0], frame[:, 1]
        
        # Plot the two orthonormal vectors
        ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], 
                  color=colors[i], alpha=0.7, arrow_length_ratio=0.1)
        ax1.quiver(0, 0, 0, v2[0], v2[1], v2[2], 
                  color=colors[i], alpha=0.7, arrow_length_ratio=0.1, linestyle='--')
        
        # Plot the plane spanned by the two vectors
        t = np.linspace(-0.5, 0.5, 10)
        s = np.linspace(-0.5, 0.5, 10)
        T, S = np.meshgrid(t, s)
        
        plane_points = T.reshape(-1, 1) * v1 + S.reshape(-1, 1) * v2
        X_plane = plane_points[:, 0].reshape(T.shape)
        Y_plane = plane_points[:, 1].reshape(T.shape)
        Z_plane = plane_points[:, 2].reshape(T.shape)
        
        ax1.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1, color=colors[i])
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')  
    ax1.set_zlabel('Z')
    ax1.set_title('V(3,2): Orthonormal 2-frames in R³')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    
    # Plot 2: Project onto Grassmannian (show spanned planes)
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i, frame in enumerate(frames[:10]):
        v1, v2 = frame[:, 0], frame[:, 1]
        
        # Create a small patch representing the 2D subspace
        t = np.array([-0.3, 0.3, 0.3, -0.3, -0.3])
        s = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
        
        patch_points = np.outer(t, v1) + np.outer(s, v2)
        
        ax2.plot(patch_points[:, 0], patch_points[:, 1], patch_points[:, 2], 
                color=colors[i], alpha=0.7, linewidth=2)
        
        # Mark the center
        ax2.scatter([0], [0], [0], color='black', s=20)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Grassmannian Gr(2,3): 2-planes through origin')
    ax2.set_xlim([-0.5, 0.5])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_zlim([-0.5, 0.5])
    
    plt.tight_layout()
    return fig


def visualize_stiefel_v41():
    """
    Visualize V(4,1) - unit sphere S³ in R⁴ (projected to 3D)
    """
    fig = plt.figure(figsize=(12, 5))
    
    # Generate points on S³
    frames = generate_random_orthonormal_frame(4, 1, 500)
    points_4d = frames[:, :, 0]  # Shape: (500, 4)
    
    # Stereographic projection from S³ to R³
    def stereographic_projection(points_4d):
        """Project S³ to R³ via stereographic projection"""
        w, x, y, z = points_4d[:, 0], points_4d[:, 1], points_4d[:, 2], points_4d[:, 3]
        
        # Avoid division by zero
        denom = 1 - w + 1e-10
        
        x_proj = x / denom
        y_proj = y / denom  
        z_proj = z / denom
        
        return np.column_stack([x_proj, y_proj, z_proj])
    
    projected_points = stereographic_projection(points_4d)
    
    # Plot 1: Stereographic projection
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Filter out points that are too far (near the "point at infinity")
    mask = np.linalg.norm(projected_points, axis=1) < 5
    filtered_points = projected_points[mask]
    
    ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
               c=np.linalg.norm(filtered_points, axis=1), cmap='plasma', 
               alpha=0.6, s=20)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('V(4,1) = S³ via Stereographic Projection')
    
    # Plot 2: Cross-sections of S³
    ax2 = fig.add_subplot(122)
    
    # Show 2D cross-sections by fixing one coordinate
    theta = np.linspace(0, 2*np.pi, 100)
    
    # w=0 cross-section (equator)
    x_eq = np.cos(theta)
    y_eq = np.sin(theta)
    ax2.plot(x_eq, y_eq, label='w=0 cross-section', linewidth=2)
    
    # w=0.5 cross-section  
    r = np.sqrt(1 - 0.5**2)
    x_05 = r * np.cos(theta)
    y_05 = r * np.sin(theta)
    ax2.plot(x_05, y_05, label='w=0.5 cross-section', linewidth=2)
    
    # w=0.8 cross-section
    r = np.sqrt(1 - 0.8**2)
    x_08 = r * np.cos(theta)
    y_08 = r * np.sin(theta)
    ax2.plot(x_08, y_08, label='w=0.8 cross-section', linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Cross-sections of S³ in R⁴')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def visualize_stiefel_v42():
    """
    Visualize V(4,2) - orthonormal 2-frames in R⁴
    Show the structure by examining projections and invariants
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Generate random orthonormal 2-frames in R⁴
    frames = generate_random_orthonormal_frame(4, 2, 200)
    
    # Plot 1: Pairwise dot products (should be zero for orthogonal vectors)
    ax1 = fig.add_subplot(131)
    
    dot_products = []
    norms_v1 = []
    norms_v2 = []
    
    for frame in frames:
        v1, v2 = frame[:, 0], frame[:, 1]
        dot_products.append(np.dot(v1, v2))
        norms_v1.append(np.linalg.norm(v1))
        norms_v2.append(np.linalg.norm(v2))
    
    ax1.hist(dot_products, bins=30, alpha=0.7, density=True)
    ax1.axvline(0, color='red', linestyle='--', label='Expected: 0')
    ax1.set_xlabel('Dot product v₁·v₂')
    ax1.set_ylabel('Density')
    ax1.set_title('Orthogonality Check')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Norms (should be 1 for unit vectors)
    ax2 = fig.add_subplot(132)
    
    # Check if we have enough variation for histogram
    all_norms = np.concatenate([norms_v1, norms_v2])
    norm_range = np.max(all_norms) - np.min(all_norms)
    
    if norm_range > 1e-6:  # If there's meaningful variation
        ax2.hist(norms_v1, bins=20, alpha=0.7, label='||v₁||', density=True)
        ax2.hist(norms_v2, bins=20, alpha=0.7, label='||v₂||', density=True)
    else:  # If norms are essentially identical
        ax2.axvline(np.mean(all_norms), color='blue', alpha=0.7, linewidth=3, label=f'||v₁||, ||v₂|| ≈ {np.mean(all_norms):.6f}')
    
    ax2.axvline(1, color='red', linestyle='--', label='Expected: 1')
    ax2.set_xlabel('Vector norm')
    ax2.set_ylabel('Density')
    ax2.set_title('Unit Vector Check')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Projection to 3D subspace
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Project first vectors to first 3 coordinates
    v1_proj = frames[:50, :3, 0]  # Take first 50 for clarity
    
    ax3.scatter(v1_proj[:, 0], v1_proj[:, 1], v1_proj[:, 2],
               alpha=0.6, s=20, c='blue', label='First vectors (3D proj)')
    
    # Add unit sphere for reference
    phi = np.linspace(0, 2*np.pi, 20)
    theta = np.linspace(0, np.pi, 10)
    PHI, THETA = np.meshgrid(phi, theta)
    
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    
    ax3.plot_wireframe(X, Y, Z, alpha=0.1, color='gray')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Projection of V(4,2) to R³')
    ax3.legend()
    
    plt.tight_layout()
    return fig


def main():
    """
    Main function to generate all Stiefel manifold visualizations
    """
    print("🔺 Generating Stiefel Manifold Visualizations...")
    print("=" * 50)
    
    print("1. V(3,1) = S² - Unit sphere in R³")
    fig1 = visualize_stiefel_v31()
    plt.show()
    
    print("\n2. V(3,2) - Orthonormal 2-frames in R³")
    fig2 = visualize_stiefel_v32()
    plt.show()
    
    print("\n3. V(4,1) = S³ - Unit sphere in R⁴")
    fig3 = visualize_stiefel_v41()
    plt.show()
    
    print("\n4. V(4,2) - Orthonormal 2-frames in R⁴")
    fig4 = visualize_stiefel_v42()
    plt.show()
    
    print("\n✅ All visualizations complete!")
    print("\nStiefel Manifold Summary:")
    print("• V(n,k) = space of orthonormal k-frames in Rⁿ")
    print("• V(3,1) = S² (2-sphere)")
    print("• V(4,1) = S³ (3-sphere)")  
    print("• V(n,k) has dimension nk - k(k+1)/2")
    print("• dim(V(3,2)) = 6 - 3 = 3")
    print("• dim(V(4,2)) = 8 - 3 = 5")


if __name__ == "__main__":
    main()
