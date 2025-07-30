import numpy as np
import csv
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import imageio
import os

frame_images = []
frame_count = 0

#to record this as an mp4
def capture_frame():
    global frame_count
    img = view.readQImage()
    img = img.convertToFormat(4)  # Format_RGBA8888
    width = img.width()
    height = img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    frame_images.append(arr)
    print(f"Captured frame {frame_count}")
    frame_count += 1

# Utility function to calculate shortest distance from a point to a 3D line segment
def point_to_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)  # projection scalar
    t = np.clip(t, 0, 1)  # clamp t to stay within segment
    closest = a + t * ab  # closest point on the segment to point p
    return np.linalg.norm(p - closest)  # return distance

# Class to represent a single integrin (with red and blue cylinders)
class Integrin:
    def __init__(self, view, id, base_position, length=0.1, radius=0.005,
                 colors=[(1, 0, 0, 1), (0, 0, 1, 1)]):
        self.view = view
        self.base_position = base_position
        self.length = length
        self.radius = radius
        self.colors = colors
        self.state = 'inactive'
        self.id = id
        self.items = []  # holds GLMeshItem graphics objects

    def add_inactive_integrin(self):
        # Add 2 subunits and rotate heads for "inactive" state
        x, y, z = self.base_position
        self.items.clear()

        # Bottom cylinders
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded', drawEdges=False)
            cyl.scale(self.radius, self.radius, self.length / 2)
            cyl.translate(x + (i - 0.5) * self.radius * 3, y, z)
            self.items.append(cyl)

        # Top rotated cylinders
        for i, color in enumerate(self.colors):
            top = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded', drawEdges=False)
            top.rotate(90, 1, 0, 0)
            top.scale(self.radius, self.radius, self.length / 2)
            top.translate(x + (i - 0.5) * self.radius * 3, y, z + self.length / 2)
            self.items.append(top)

        self.view_integrin()

    def add_active_integrin(self):
        # Draw single straight red and blue cylinders when active
        x, y, z = self.base_position
        self.items.clear()

        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded', drawEdges=False)
            cyl.scale(self.radius, self.radius, self.length)
            cyl.translate(x + (i - 0.5) * self.radius * 3, y, z)
            self.items.append(cyl)

        self.view_integrin()

    def view_integrin(self):
        # Add each mesh to view
        for item in self.items:
            self.view.addItem(item)

    def remove_integrin(self):
        # Remove mesh objects from view
        for item in self.items:
            self.view.removeItem(item)
        self.items = []

    def switch_to_active(self):
        # Transition from inactive to active state
        if self.state == 'inactive':
            self.remove_integrin()
            self.add_active_integrin()
            self.state = 'active'

    def get_tip_position(self):
        # Return the tip of the integrin (for distance check)
        x, y, z = self.base_position
        return np.array([x, y, z + self.length])

# Class to manage a group of integrins
class IntegrinManager:
    def __init__(self, view):
        self.view = view
        self.integrins = {}

    def populate_integrins(self, N, radius, z=0):
        # Generate integrins in a spiral lattice on a circle
        positions = self._fibonacci_lattice_positions(N, radius, z)
        for i, pos in enumerate(positions):
            integrin = Integrin(self.view, id=f"integrin{i+1}", base_position=pos)
            integrin.add_inactive_integrin()
            self.integrins[integrin.id] = integrin

    def _fibonacci_lattice_positions(self, N, R, z):
        # Return evenly distributed N points on a disc
        golden_angle = 2.3999632297
        positions = []
        for i in range(N):
            r = R * np.sqrt(i / N)
            theta = i * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append((x, y, z))
        return positions

    def activate_near_fiber(self, fiber_start, fiber_end, threshold=0.05, save_path="integrin_distances.csv"):
        any_activated = False
        data = []

        for integrin in self.integrins.values():
            tip = integrin.get_tip_position()
            dist = point_to_segment_distance(tip, fiber_start, fiber_end)
            activated = False
            if dist < threshold:
                integrin.switch_to_active()
                activated = True
                any_activated = True
            data.append([integrin.id, tip[0], tip[1], tip[2], dist, activated])

        # Save to CSV
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Tip_X', 'Tip_Y', 'Tip_Z', 'Distance_to_Fiber', 'Activated'])
            writer.writerows(data)

        return any_activated

# Fiber class representing a rigid ECM fiber
class Fiber:
    def __init__(self, view, start, end, radius=0.02, color=(0.5, 0.5, 0.5, 1)):
        self.view = view
        self.start = np.array(start)
        self.end = np.array(end)
        self.radius = radius
        self.color = color
        self.item = None

    def draw(self):
        # Draw a 3D cylinder from start to end
        direction = self.end - self.start
        height = np.linalg.norm(direction)
        if height == 0:
            return

        direction /= height
        center = (self.start + self.end) / 2
        mesh = gl.MeshData.cylinder(rows=10, cols=20)
        self.item = gl.GLMeshItem(meshdata=mesh, smooth=True, color=self.color, shader='shaded', drawEdges=False)
        self.item.scale(self.radius, self.radius, height)

        # Rotate to align with direction vector
        default_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(default_axis, direction)
        angle = np.degrees(np.arccos(np.dot(default_axis, direction)))
        if np.linalg.norm(rotation_axis) > 1e-6:
            self.item.rotate(angle, *rotation_axis)

        self.item.translate(*center)
        self.view.addItem(self.item)

    def pull_towards(self, z_target, step=0.02):
        # Move fiber closer to cell plane
        self.view.removeItem(self.item)
        midpoint_z = (self.start[2] + self.end[2]) / 2
        if midpoint_z > z_target:
            self.start[2] -= step
            self.end[2] -= step
        self.draw()

# ---------------- MAIN SIMULATION -----------------

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.setWindowTitle('Integrin Activation and Fiber Pulling')
view.setGeometry(0, 110, 1280, 800)
view.show()

# Create integrins on plane z=0
manager = IntegrinManager(view)
manager.populate_integrins(N=5, radius=0.85, z=0)

# Create random fiber position in x, y, and z
fiber = Fiber(view,
              start=(-0.5, -0.7, 0.14),  # left side of the integrin
              end=(0.6, 0.6, 0.14))     # right side of the integrin

# Frame 1: Wait
def frame_1():
    print("Frame 1: Waiting...")
    capture_frame()

# Frame 2: Draw fiber and activate integrins
def frame_2():
    print("Frame 2: Drawing fiber and activating integrins...")
    fiber.draw()
    global any_activated
    any_activated = manager.activate_near_fiber(fiber_start=fiber.start, fiber_end=fiber.end, threshold=0.11)
    capture_frame()

# Frame 3: Only pull fiber if integrins were activated
def frame_3():
    global any_activated
    print("Frame 3: Pulling fiber...")
    if any_activated:
        fiber.pull_towards(z_target=0.02)
    else:
        print("No integrins close enough — fiber stays put.")
    capture_frame()
    save_video()

#function to save video
def save_video():
    output_filename = "integrin_simulation.mp4"
    print(f"Saving video to {output_filename}...")
    imageio.mimsave(output_filename, frame_images, fps=1)  # Adjust fps if needed

# Schedule simulation
QtCore.QTimer.singleShot(1000, frame_1)
QtCore.QTimer.singleShot(2500, frame_2)
QtCore.QTimer.singleShot(4000, frame_3)

app.exec_()
