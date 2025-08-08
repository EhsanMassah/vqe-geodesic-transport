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


def stereographic_projection(points_nd: np.ndarray) -> np.ndarray:
    """
    Apply stereographic projection from S^(n-1) to R^(n-1).
    Projects from the north pole to the hyperplane.
    
    Parameters:
    -----------
    points_nd : np.ndarray
        Points on the n-dimensional unit sphere, shape (num_points, n)
        
    Returns:
    --------
    np.ndarray
        Projected points in (n-1)-dimensional space, shape (num_points, n-1)
    """
    # Use the first coordinate as the projection coordinate
    w = points_nd[:, 0]
    coords = points_nd[:, 1:]
    
    # Avoid division by zero (points near north pole)
    denom = 1 - w + 1e-10
    
    # Project to hyperplane
    projected = coords / denom.reshape(-1, 1)
    
    return projected


def plot_sphere_surface(ax, dimension: int = 3):
    """
    Plot the surface of a unit sphere in the given dimension (2D or 3D visualization).
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    dimension : int
        Dimension of the sphere (2 for circle, 3 for sphere surface)
    """
    if dimension == 2:
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, 'b-', alpha=0.3, linewidth=2)
        
    elif dimension == 3:
        # Plot unit sphere surface
        phi = np.linspace(0, 2*np.pi, 30)
        theta = np.linspace(0, np.pi, 15)
        PHI, THETA = np.meshgrid(phi, theta)
        
        X = np.sin(THETA) * np.cos(PHI)
        Y = np.sin(THETA) * np.sin(PHI) 
        Z = np.cos(THETA)
        
        ax.plot_surface(X, Y, Z, alpha=0.2, cmap='viridis')


def compute_frame_statistics(frames: np.ndarray, k: int) -> dict:
    """
    Compute validation statistics for orthonormal frames.
    
    Parameters:
    -----------
    frames : np.ndarray
        Array of orthonormal frames
    k : int
        Number of vectors per frame
        
    Returns:
    --------
    dict
        Dictionary containing dot products, norms, and other statistics
    """
    all_dot_products = []
    all_norms = []
    
    for frame in frames:
        vectors = [frame[:, i] for i in range(k)]
        
        # Compute pairwise dot products
        for i in range(k):
            for j in range(i + 1, k):
                all_dot_products.append(np.dot(vectors[i], vectors[j]))
            all_norms.append(np.linalg.norm(vectors[i]))
    
    return {
        'dot_products': all_dot_products,
        'norms': all_norms,
        'mean_dot_product': np.mean(all_dot_products) if all_dot_products else 0,
        'std_dot_product': np.std(all_dot_products) if all_dot_products else 0,
        'mean_norm': np.mean(all_norms),
        'std_norm': np.std(all_norms),
        'max_abs_dot_product': np.max(np.abs(all_dot_products)) if all_dot_products else 0
    }


def create_info_text(n: int, k: int, stats: dict, num_frames: int, 
                    extra_info: str = "") -> str:
    """Create standardized info text for validation displays."""
    manifold_dim = n * k - k * (k + 1) // 2
    
    base_text = f"""
Stiefel Manifold V({n},{k})

Geometry:
• Ambient space: R^{n}
• Frame size: {k} vectors
• Manifold dimension: {manifold_dim}

Samples: {num_frames} frames

Orthogonality Check:
• Mean dot product: {stats['mean_dot_product']:.2e}
• Max |dot product|: {stats['max_abs_dot_product']:.2e}

Unit Norm Check:
• Mean norm: {stats['mean_norm']:.6f}
• Norm std: {stats['std_norm']:.2e}
"""
    
    if extra_info:
        base_text += f"\n{extra_info}"
    
    return base_text


def visualize_orthogonality_checks(frames: np.ndarray, n: int, k: int) -> Tuple[plt.Figure, dict]:
    """Create validation plots for orthonormal frames."""
    stats = compute_frame_statistics(frames, k)
    manifold_dim = n * k - k * (k + 1) // 2
    stats['manifold_dim'] = manifold_dim
    
    num_plots = min(3, k + 1)
    fig = plt.figure(figsize=(5 * num_plots, 4))
    plot_idx = 1
    
    # Plot 1: Orthogonality check (if k > 1)
    if k > 1:
        ax1 = fig.add_subplot(1, num_plots, plot_idx)
        ax1.hist(stats['dot_products'], bins=25, alpha=0.7, density=True, color='skyblue')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        ax1.set_xlabel('Pairwise dot products')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Orthogonality Check\nV({n},{k})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 2: Unit norm check
    if plot_idx <= num_plots:
        ax2 = fig.add_subplot(1, num_plots, plot_idx)
        
        norm_range = np.max(stats['norms']) - np.min(stats['norms'])
        if norm_range > 1e-6:
            ax2.hist(stats['norms'], bins=20, alpha=0.7, density=True, color='lightgreen')
        else:
            ax2.axvline(stats['mean_norm'], color='blue', alpha=0.7, linewidth=4,
                       label=f'||v|| ≈ {stats["mean_norm"]:.6f}')
        
        ax2.axvline(1, color='red', linestyle='--', linewidth=2, label='Expected: 1')
        ax2.set_xlabel('Vector norms')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Unit Norm Check\nV({n},{k})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 3: Info panel
    if plot_idx <= num_plots:
        ax3 = fig.add_subplot(1, num_plots, plot_idx)
        ax3.axis('off')
        info_text = create_info_text(n, k, stats, len(frames), 
                                   "Formula: dim = nk - k(k+1)/2")
        ax3.text(0.1, 0.9, info_text, fontsize=10, fontfamily='monospace',
                verticalalignment='top', transform=ax3.transAxes)
    
    plt.tight_layout()
    return fig, stats


def visualize_stiefel_manifold(n: int, k: int, num_samples: int = 200,
                              visualization_type: str = 'auto') -> plt.Figure:
    """
    General visualization function for Stiefel manifolds V(n,k).
    
    Parameters:
    -----------
    n : int
        Dimension of ambient space (2 ≤ n ≤ 4)
    k : int
        Number of orthonormal vectors in each frame (1 ≤ k ≤ n)
    num_samples : int
        Number of random frames to generate
    visualization_type : str
        Type of visualization: 'auto', 'geometric', 'validation', 'projection'
        
    Returns:
    --------
    plt.Figure
        The generated figure
        
    Raises:
    -------
    ValueError
        If parameters are invalid
    """
    # Validate inputs
    if not (2 <= n <= 4):
        raise ValueError("Ambient dimension n must be between 2 and 4")
    if not (1 <= k <= n):
        raise ValueError(f"Frame size k must be between 1 and {n}")
    
    # Generate random orthonormal frames
    frames = generate_random_orthonormal_frame(n, k, num_samples)
    
    # Determine visualization strategy
    manifold_dim = n * k - k * (k + 1) // 2
    
    if visualization_type == 'validation':
        return visualize_orthogonality_checks(frames, n, k)[0]
    
    # Main geometric visualization
    if n <= 3 and k == 1:
        # Unit sphere case - can visualize directly
        return _visualize_unit_sphere(frames, n)
        
    elif n == 3 and k == 2:
        # 2-frames in R³ - show vector pairs and Grassmannian
        return _visualize_frames_3d(frames, n, k)
        
    elif n == 4 and k == 1:
        # S³ in R⁴ - use projections
        return _visualize_sphere_4d(frames)
        
    elif n == 4 and k >= 2:
        # Higher-dimensional frames - show projections and validation
        return _visualize_high_dimensional_frames(frames, n, k)
        
    else:
        # General case - validation and projection plots
        return _visualize_general_case(frames, n, k)


def _visualize_unit_sphere(frames: np.ndarray, n: int) -> plt.Figure:
    """Visualize V(n,1) = S^(n-1) for n ≤ 3"""
    points = frames[:, :, 0]  # Extract single vectors
    
    if n == 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot unit circle
        plot_sphere_surface(ax, dimension=2)
        
        # Plot random points
        ax.scatter(points[:, 0], points[:, 1], c='red', s=30, alpha=0.8,
                  label=f'Random points on V(2,1) = S¹')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Stiefel Manifold V(2,1) = S¹ (Unit Circle)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    else:  # n == 3
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot unit sphere surface
        plot_sphere_surface(ax, dimension=3)
        
        # Plot random points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='red', s=50, alpha=0.8, label=f'Random points on V(3,1)')
        
        # Add coordinate axes
        ax.quiver([0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,1,0], [0,0,1],
                 color=['red', 'green', 'blue'], alpha=0.6, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Stiefel Manifold V(3,1) = S² (Unit Sphere)')
        ax.legend()
    
    plt.tight_layout()
    return fig


def _visualize_frames_3d(frames: np.ndarray, n: int, k: int) -> plt.Figure:
    """Visualize k-frames in R³"""
    fig = plt.figure(figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(frames)))
    
    # Plot 1: Vector visualization
    ax1 = fig.add_subplot(131, projection='3d')
    
    display_frames = frames[:min(8, len(frames))]  # Show subset for clarity
    for i, frame in enumerate(display_frames):
        for j in range(k):
            vector = frame[:, j]
            linestyle = '-' if j == 0 else '--'
            ax1.quiver(0, 0, 0, vector[0], vector[1], vector[2],
                      color=colors[i], alpha=0.7, arrow_length_ratio=0.1,
                      linestyle=linestyle)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'V({n},{k}): {k}-frames in R³')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    
    # Plot 2: Subspace visualization (if k >= 2)
    if k >= 2:
        ax2 = fig.add_subplot(132, projection='3d')
        
        for i, frame in enumerate(display_frames):
            # Create small patch representing the k-dimensional subspace
            if k == 2:
                # For 2-frames, show the spanned plane
                t = np.linspace(-0.4, 0.4, 5)
                s = np.linspace(-0.4, 0.4, 5)
                T, S = np.meshgrid(t, s)
                
                v1, v2 = frame[:, 0], frame[:, 1]
                plane_points = T.reshape(-1, 1) * v1 + S.reshape(-1, 1) * v2
                X_plane = plane_points[:, 0].reshape(T.shape)
                Y_plane = plane_points[:, 1].reshape(T.shape)
                Z_plane = plane_points[:, 2].reshape(T.shape)
                
                ax2.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color=colors[i])
            
            elif k == 3:
                # For 3-frames, show the spanned 3D space (which is all of R³)
                # Just show the three basis vectors
                for j in range(k):
                    vector = frame[:, j]
                    ax2.quiver(0, 0, 0, vector[0], vector[1], vector[2],
                              color=colors[i], alpha=0.7, arrow_length_ratio=0.1)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'Subspaces spanned by {k}-frames')
        ax2.set_xlim([-0.5, 0.5])
        ax2.set_ylim([-0.5, 0.5])
        ax2.set_zlim([-0.5, 0.5])
    
    # Plot 3: Validation info panel
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    stats = compute_frame_statistics(frames, k)
    info_text = create_info_text(n, k, stats, len(frames))
    ax3.text(0.1, 0.9, info_text, fontsize=9, fontfamily='monospace',
            verticalalignment='top', transform=ax3.transAxes)
    
    plt.tight_layout()
    return fig


def plot_coordinate_projections(ax, points_4d: np.ndarray, title: str):
    """Plot 2D coordinate projections for 4D points."""
    # Project to different coordinate plane pairs
    projections = [
        (points_4d[:, [0, 1]], '(x,y)'),
        (points_4d[:, [2, 3]], '(z,w)')
    ]
    
    for proj_data, label in projections:
        ax.scatter(proj_data[:, 0], proj_data[:, 1], alpha=0.5, s=10, label=f'{label} projection')
    
    # Add reference unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x, circle_y = np.cos(theta), np.sin(theta)
    ax.plot(circle_x, circle_y, 'k--', alpha=0.3, label='Unit circle')
    
    ax.set_xlabel('First coordinate')
    ax.set_ylabel('Second coordinate')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_coordinate_distributions(ax, points_4d: np.ndarray, title: str):
    """Plot coordinate distributions for 4D points."""
    coords = ['x', 'y', 'z', 'w']
    for i, coord in enumerate(coords):
        ax.hist(points_4d[:, i], bins=20, alpha=0.5, label=f'{coord} coordinate', density=True)
    
    ax.set_xlabel('Coordinate value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _visualize_sphere_4d(frames: np.ndarray) -> plt.Figure:
    """Visualize S³ in R⁴ using projections"""
    points_4d = frames[:, :, 0]  # Extract single vectors
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Stereographic projection to R³
    ax1 = fig.add_subplot(131, projection='3d')
    
    projected_points = stereographic_projection(points_4d)
    
    # Filter out points too far from origin
    mask = np.linalg.norm(projected_points, axis=1) < 5
    filtered_points = projected_points[mask]
    
    ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
               c=np.linalg.norm(filtered_points, axis=1), cmap='plasma',
               alpha=0.6, s=20)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('V(4,1) = S³\nStereographic Projection')
    
    # Plot 2: 2D coordinate projections
    ax2 = fig.add_subplot(132)
    plot_coordinate_projections(ax2, points_4d, '2D Coordinate Projections')
    
    # Plot 3: Coordinate distributions
    ax3 = fig.add_subplot(133)
    plot_coordinate_distributions(ax3, points_4d, 'Coordinate Distributions on S³')
    
    plt.tight_layout()
    return fig
    """Visualize S³ in R⁴ using projections"""
    points_4d = frames[:, :, 0]  # Extract single vectors
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Stereographic projection to R³
    ax1 = fig.add_subplot(131, projection='3d')
    
    projected_points = stereographic_projection(points_4d)
    
    # Filter out points too far from origin
    mask = np.linalg.norm(projected_points, axis=1) < 5
    filtered_points = projected_points[mask]
    
    ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
               c=np.linalg.norm(filtered_points, axis=1), cmap='plasma',
               alpha=0.6, s=20)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('V(4,1) = S³\nStereographic Projection')
    
def _visualize_high_dimensional_frames(frames: np.ndarray, n: int, k: int) -> plt.Figure:
    """Visualize high-dimensional frames using projections and validation"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Project first vectors to 3D
    ax1 = fig.add_subplot(131, projection='3d')
    
    if n >= 3:
        proj_3d = frames[:50, :3, 0]  # Take first 50 samples, project to 3D
        ax1.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2],
                   alpha=0.6, s=20, c='blue')
        
        # Add unit sphere for reference
        plot_sphere_surface(ax1, dimension=3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'V({n},{k}): 3D Projection\nof First Vectors')
    
    # Plot 2: Pairwise distances between frames
    ax2 = fig.add_subplot(132)
    
    # Compute Frobenius distances between frames
    sample_frames = frames[:50]  # Sample for computational efficiency
    distances = []
    
    for i in range(len(sample_frames)):
        for j in range(i + 1, len(sample_frames)):
            frame_diff = sample_frames[i] - sample_frames[j]
            dist = np.linalg.norm(frame_diff, 'fro')
            distances.append(dist)
    
    ax2.hist(distances, bins=20, alpha=0.7, density=True, color='orange')
    ax2.set_xlabel('Frobenius distance between frames')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Inter-frame Distances\nV({n},{k})')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation info panel
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    stats = compute_frame_statistics(frames, k)
    extra_info = f"Sample Statistics:\n• Mean inter-frame distance: {np.mean(distances):.3f}"
    info_text = create_info_text(n, k, stats, len(frames), extra_info)
    ax3.text(0.1, 0.9, info_text, fontsize=9, fontfamily='monospace',
            verticalalignment='top', transform=ax3.transAxes)
    
    plt.tight_layout()
    return fig


def _visualize_general_case(frames: np.ndarray, n: int, k: int) -> plt.Figure:
    """General visualization for any V(n,k) - primarily validation plots"""
    return visualize_orthogonality_checks(frames, n, k)[0]


# Backward compatibility - simple aliases
visualize_stiefel_v31 = lambda: visualize_stiefel_manifold(3, 1)
visualize_stiefel_v32 = lambda: visualize_stiefel_manifold(3, 2)
visualize_stiefel_v41 = lambda: visualize_stiefel_manifold(4, 1)
visualize_stiefel_v42 = lambda: visualize_stiefel_manifold(4, 2)


def main():
    """
    Main function to demonstrate the general Stiefel manifold visualization
    """
    print("🔺 General Stiefel Manifold Visualization System")
    print("=" * 55)
    
    # Examples of different manifolds
    examples = [
        (2, 1, "V(2,1) = S¹ - Unit circle in R²"),
        (3, 1, "V(3,1) = S² - Unit sphere in R³"),
        (3, 2, "V(3,2) - Orthonormal 2-frames in R³"),
        (4, 1, "V(4,1) = S³ - Unit sphere in R⁴"),
        (4, 2, "V(4,2) - Orthonormal 2-frames in R⁴"),
        (4, 3, "V(4,3) - Orthonormal 3-frames in R⁴")
    ]
    
    for i, (n, k, description) in enumerate(examples, 1):
        print(f"\n{i}. {description}")
        
        try:
            fig = visualize_stiefel_manifold(n, k, num_samples=150)
            plt.show()
            
            # Show validation for higher-dimensional cases
            if n * k > 6:  # For complex cases, also show validation
                print(f"   → Validation plots for V({n},{k})")
                validation_fig = visualize_stiefel_manifold(n, k, num_samples=200, 
                                                          visualization_type='validation')
                plt.show()
                
        except Exception as e:
            print(f"   ❌ Error visualizing V({n},{k}): {e}")
    
    print("\n✅ All visualizations complete!")
    print("\n" + "=" * 55)
    print("General Stiefel Manifold Function Usage:")
    print("• visualize_stiefel_manifold(n, k, num_samples=200)")
    print("• Parameters: n (ambient dim), k (frame size)")
    print("• Supports: 2 ≤ n ≤ 4, 1 ≤ k ≤ n")
    print("• Automatic visualization strategy selection")
    print("• Types: 'auto', 'geometric', 'validation', 'projection'")
    print("\nDimensions computed: dim(V(n,k)) = nk - k(k+1)/2")
    
    # Print dimension table
    print("\nDimension Table:")
    print("Manifold    | Dimension")
    print("------------|----------")
    for n, k, desc in examples:
        dim = n * k - k * (k + 1) // 2
        print(f"V({n},{k})       | {dim}")


def demo_interactive():
    """
    Interactive demo function for testing different parameters
    """
    print("\n🔺 Interactive Stiefel Manifold Explorer")
    print("Enter parameters to visualize V(n,k)")
    print("Valid ranges: 2 ≤ n ≤ 4, 1 ≤ k ≤ n")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter n,k (e.g., '3,2') or 'quit': ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
                
            n, k = map(int, user_input.split(','))
            
            print(f"\n🔹 Generating V({n},{k}) visualization...")
            fig = visualize_stiefel_manifold(n, k)
            plt.show()
            
            manifold_dim = n * k - k * (k + 1) // 2
            print(f"✅ V({n},{k}) has dimension {manifold_dim}")
            
        except ValueError:
            print("❌ Invalid input format. Use 'n,k' format (e.g., '3,2')")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        demo_interactive()
    else:
        main()
