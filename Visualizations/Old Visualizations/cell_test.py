
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fibonacci_sphere(samples, radius, min_distance):
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(samples * 10):  # generate more to allow for rejection sampling
        y = 1 - (i / float(samples - 1)) * 2
        y = np.clip(y, -1.0, 1.0)
        radius_xy = np.sqrt(max(0, 1 - y**2))  # ensure no negative sqrt
        theta = phi * i
        x = np.cos(theta) * radius_xy
        z = np.sin(theta) * radius_xy
        new_point = (x * radius, y * radius, z * radius)

        # Reject if too close to existing points
        too_close = any(np.linalg.norm(np.array(new_point) - np.array(p)) < min_distance for p in points)
        if not too_close:
            points.append(new_point)
        if len(points) >= samples:
            break

    return points
def distribute_integrin_clusters(fa_center, fa_radius, num_clusters):
    clusters = []
    phi = (1 + np.sqrt(5)) / 2
    for i in range(num_clusters):
        r = fa_radius * np.sqrt(i / num_clusters)
        theta = 2 * np.pi * i / phi
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)

        norm = np.linalg.norm(fa_center)
        nx, ny, nz = fa_center / norm
        if nx == 0 and ny == 0:
            perp_x = np.array([1, 0, 0])
        else:
            perp_x = np.cross([0, 0, 1], [nx, ny, nz])
            perp_x /= np.linalg.norm(perp_x)
        perp_y = np.cross([nx, ny, nz], perp_x)
        perp_y /= np.linalg.norm(perp_y)

        new_x = fa_center[0] + dx * perp_x[0] + dy * perp_y[0]
        new_y = fa_center[1] + dx * perp_x[1] + dy * perp_y[1]
        new_z = fa_center[2] + dx * perp_x[2] + dy * perp_y[2]

        norm = np.linalg.norm([new_x, new_y, new_z])
        new_x = (new_x / norm) * cell_radius
        new_y = (new_y / norm) * cell_radius
        new_z = (new_z / norm) * cell_radius
        clusters.append((new_x, new_y, new_z))
    return clusters

def distribute_integrins(cluster_center, cluster_radius, num_integrins):
    integrins = []
    phi = (1 + np.sqrt(5)) / 2
    for i in range(num_integrins):
        r = cluster_radius * np.sqrt(i / num_integrins)
        theta = 2 * np.pi * i / phi
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)

        norm = np.linalg.norm(cluster_center)
        nx, ny, nz = cluster_center / norm
        if nx == 0 and ny == 0:
            perp_x = np.array([1, 0, 0])
        else:
            perp_x = np.cross([0, 0, 1], [nx, ny, nz])
            perp_x /= np.linalg.norm(perp_x)
        perp_y = np.cross([nx, ny, nz], perp_x)
        perp_y /= np.linalg.norm(perp_y)

        new_x = cluster_center[0] + dx * perp_x[0] + dy * perp_y[0]
        new_y = cluster_center[1] + dx * perp_x[1] + dy * perp_y[1]
        new_z = cluster_center[2] + dx * perp_x[2] + dy * perp_y[2]

        norm = np.linalg.norm([new_x, new_y, new_z])
        new_x = (new_x / norm) * cell_radius
        new_y = (new_y / norm) * cell_radius
        new_z = (new_z / norm) * cell_radius
        integrins.append((new_x, new_y, new_z))
    return integrins

# Parameters
cell_radius = 12
num_focal_adhesions = 65
cluster_radius = 2
min_fa_distance = 2*cluster_radius
num_clusters_per_fa = 10
num_integrins_per_cluster = 2
fa_radius = 3

# Generate positions
focal_adhesions = fibonacci_sphere(num_focal_adhesions, cell_radius, min_fa_distance)
integrin_clusters = []
integrins = []

for fa in focal_adhesions:
    clusters = distribute_integrin_clusters(fa, cluster_radius, num_clusters_per_fa)
    integrin_clusters.extend(clusters)
    for cluster in clusters:
        integrins.extend(distribute_integrins(cluster, cluster_radius, num_integrins_per_cluster))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Simulated Cell with FAs (Red), Clusters (Orange), Integrins (Blue)")

focal_adhesions = np.array(focal_adhesions)
clusters = np.array(integrin_clusters)
integrins = np.array(integrins)

ax.scatter(focal_adhesions[:,0], focal_adhesions[:,1], focal_adhesions[:,2], color='red', s=60, label='Focal Adhesions')
ax.scatter(clusters[:,0], clusters[:,1], clusters[:,2], color='orange', s=30, label='Integrin Clusters')
ax.scatter(integrins[:,0], integrins[:,1], integrins[:,2], color='blue', s=5, label='Integrins')

ax.legend()
plt.show()
