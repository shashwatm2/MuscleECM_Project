import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---- Load Vertices ----
def load_vertices(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    vertices = [list(map(float, line.strip().split()[:3])) for line in lines]
    return np.array(vertices)

# ---- Load Edges ----
def load_edges(file_path, num_vertices):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    edges = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            i, j = int(parts[0]), int(parts[1])
            if i < num_vertices and j < num_vertices:
                edges.append((i, j))
    return edges

# ---- Apply Simulated Deformation from Cell Forces ----
def apply_deformation(vertices, cell_position, threshold=0.3, displacement_magnitude=0.55):
    deformed = vertices.copy()
    for i, v in enumerate(vertices):
        dist = np.linalg.norm(v - cell_position)
        if dist < threshold:
            direction = (v - cell_position) / (dist + 1e-6)  # displace outward from cell
            displacement = direction * displacement_magnitude
            deformed[i] += displacement
    return deformed

# ---- Visualization ----
def visualize(original_vertices, deformed_vertices, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original network (blue)
    for i, j in edges:
        x = [original_vertices[i][0], original_vertices[j][0]]
        y = [original_vertices[i][1], original_vertices[j][1]]
        z = [original_vertices[i][2], original_vertices[j][2]]
        ax.plot(x, y, z, color='blue', linewidth=2)

    # Plot deformed network (green dashed)
    for i, j in edges:
        x = [deformed_vertices[i][0], deformed_vertices[j][0]]
        y = [deformed_vertices[i][1], deformed_vertices[j][1]]
        z = [deformed_vertices[i][2], deformed_vertices[j][2]]
        ax.plot(x, y, z, color='green', linestyle='--', linewidth=2)

    # Plot vertices
    ax.scatter(original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2], color='red', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Original (blue) vs Deformed (green) ECM Fibers")
    plt.show()

# ---- Main ----
if __name__ == "__main__":
    vertices_file = 'output_input_generator/test-file_vertices.out'
    edges_file = 'output_input_generator/test-file_nodes_to_edges.out'

    original_vertices = load_vertices(vertices_file)
    edges = load_edges(edges_file,len(original_vertices))

    cell_position = np.array([0.0, 0.0, 0.0])  # Example cell position in center
    deformed_vertices = apply_deformation(original_vertices, cell_position)

    visualize(original_vertices, deformed_vertices, edges)
