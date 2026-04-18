import numpy as np
import csv
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import imageio

# -------------------- Frame capture --------------------

frame_images = []
frame_count = 0

def capture_frame():
    global frame_count
    img = view.readQImage()
    img = img.convertToFormat(4)  # Format_RGBA8888
    width, height = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    frame_images.append(arr)
    print(f"Captured frame {frame_count}")
    frame_count += 1

# -------------------- Geometry helper --------------------

def point_to_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

# -------------------- Integrin + Manager --------------------

class Integrin:
    def __init__(self, view, id, base_position, length=0.00004*1000, radius=0.000005*1000, colors=[(1, 0, 0, 1), (0, 0, 1, 1)]):
        self.view = view
        self.id = id
        self.base_position = base_position
        self.length = length
        self.radius = radius
        self.colors = colors
        self.state = 'inactive'
        self.items = []

    def add_inactive_integrin(self):
        self.items.clear()
        x, y, z = self.base_position
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20), smooth=True, color=color, shader='shaded')
            cyl.scale(self.radius, self.radius, self.length / 2)
            cyl.translate(x + (i - 0.5) * self.radius * 3, y, z)
            self.items.append(cyl)
        for i, color in enumerate(self.colors):
            top = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20), smooth=True, color=color, shader='shaded')
            top.rotate(90, 1, 0, 0)
            top.scale(self.radius, self.radius, self.length / 2)
            top.translate(x + (i - 0.5) * self.radius * 3, y, z + self.length / 2)
            self.items.append(top)
        self.view_integrin()

    def add_active_integrin(self):
        self.items.clear()
        x, y, z = self.base_position
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20), smooth=True, color=color, shader='shaded')
            cyl.scale(self.radius, self.radius, self.length)
            cyl.translate(x + (i - 0.5) * self.radius * 3, y, z)
            self.items.append(cyl)
        self.view_integrin()

    def view_integrin(self):
        for item in self.items:
            self.view.addItem(item)

    def remove_integrin(self):
        for item in self.items:
            self.view.removeItem(item)
        self.items = []

    def switch_to_active(self):
        if self.state == 'inactive':
            self.remove_integrin()
            self.add_active_integrin()
            self.state = 'active'

    def get_tip_position(self):
        x, y, z = self.base_position
        return np.array([x, y, z + self.length])

class IntegrinManager:
    def __init__(self, view):
        self.view = view
        self.integrins = {}

    def populate_integrins(self, N, radius, z=0):
        golden_angle = 2.3999632297
        for i in range(N):
            r = radius * np.sqrt(i / N)
            theta = i * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            integrin = Integrin(self.view, f"integrin{i+1}", (x, y, z))
            integrin.add_inactive_integrin()
            self.integrins[integrin.id] = integrin

    def activate_near_fiber(self, fiber_start, fiber_end, threshold=0.05, save_path="integrin_distances.csv"):
        any_activated = False
        data=[]
        for integrin in self.integrins.values():
            tip = integrin.get_tip_position()
            activated = False
            dist = point_to_segment_distance(tip, fiber_start, fiber_end)
            if dist < threshold:
                integrin.switch_to_active()
                activated = True
                any_activated = True
            data.append([integrin.id, tip[0], tip[1], tip[2], dist, activated])

        print("Activated integrins:", [i.id for i in self.integrins.values() if i.state == 'active'])
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Tip_X', 'Tip_Y', 'Tip_Z', 'Distance_to_Fiber', 'Activated'])
            writer.writerows(data)
        return any_activated

# -------------------- Elastic Fiber --------------------

class ElasticBeamFiber:
    def __init__(self, view, start, end, num_segments=20, youngs_modulus=1.1e3, radius=0.0001*1000):
        self.view = view
        self.radius = radius
        self.youngs_modulus = youngs_modulus
        self.num_segments = num_segments
        self.nodes = np.linspace(start, end, num_segments + 1)
        self.rest_lengths = [np.linalg.norm(self.nodes[i+1] - self.nodes[i]) for i in range(num_segments)]
        self.visuals = []

    def draw(self):
        for item in self.visuals:
            self.view.removeItem(item)
        self.visuals = []
        for i in range(len(self.nodes) - 1):
            seg = self._draw_segment(self.nodes[i], self.nodes[i+1])
            self.view.addItem(seg)
            self.visuals.append(seg)

    def apply_forces(self, integrin_tip_positions, F_SF=1e-3, step_scale=50):
        if not integrin_tip_positions:
            return
        A = np.pi * self.radius**2
        L = np.mean(self.rest_lengths)
        k = self.youngs_modulus * A / L
        F_per = F_SF / len(integrin_tip_positions)

        forces = [np.zeros(3) for _ in self.nodes]
        for tip in integrin_tip_positions:
            distances = [np.linalg.norm(tip - n) for n in self.nodes]
            idx = np.argmin(distances)
            direction = tip - self.nodes[idx] 
            if np.linalg.norm(direction) == 0:
                continue
            unit = direction / np.linalg.norm(direction)
            forces[idx] += F_per * unit

        for i in range(1, len(self.nodes)-1):
            self.nodes[i] += (forces[i] / k) * step_scale
        self.draw()

    def _draw_segment(self, a, b, color=(0.6, 0.6, 0.6, 1)):
        direction = b - a
        height = np.linalg.norm(direction)
        if height == 0: return
        direction /= height
        center = (a + b) / 2
        mesh = gl.MeshData.cylinder(rows=10, cols=20)
        item = gl.GLMeshItem(meshdata=mesh, smooth=True, color=color, shader='shaded', drawEdges=False)
        item.scale(self.radius, self.radius, height)
        default_axis = np.array([0, 0, 1])
        rot_axis = np.cross(default_axis, direction)
        angle = np.degrees(np.arccos(np.clip(np.dot(default_axis, direction), -1.0, 1.0)))
        if np.linalg.norm(rot_axis) > 1e-6:
            item.rotate(angle, *rot_axis)
        item.translate(*center)
        return item

# -------------------- PyQt + Simulation --------------------

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.setWindowTitle('Integrin Activation and Elastic Fiber Deformation')
view.setGeometry(0, 110, 1280, 800)
view.show()
view.setCameraPosition(distance=2.0, elevation=20, azimuth=45)

manager = IntegrinManager(view)
manager.populate_integrins(N=100, radius=0.8, z=0)

fiber = ElasticBeamFiber(view,
    start=(-0.6, -0.6, 0.12),
    end=(0.6, 0.6, 0.12),
    num_segments=50,
    youngs_modulus=1.1e3,
    radius=0.0001*75
)

def frame_1():
    print("Frame 1: Waiting...")
    capture_frame()

def frame_2():
    global any_activated
    print("Frame 2: Drawing fiber and activating integrins...")
    fiber.draw()
    any_activated = manager.activate_near_fiber(
        fiber_start=fiber.nodes[0],
        fiber_end=fiber.nodes[-1],
        threshold=0.11,
        save_path="integrin_distances.csv"
    )
    capture_frame()

def frame_3():
    print("Frame 3: Pulling fiber...")
    global any_activated
    # Log BEFORE pulling
    #with open("fiber_pull_log.csv", mode="a", newline="") as file:
      #   writer = csv.writer(file)
     #    writer.writerow(["BEFORE"])
     #    for node in fiber.nodes:
     #        writer.writerow(node.tolist())

    # Apply pulling forces
    fiber.apply_forces(
        integrin_tip_positions=[
            i.get_tip_position()
            for i in manager.integrins.values()
            if i.state == 'active'
            ]
        )

    # Log AFTER pulling
    #with open("fiber_pull_log.csv", mode="a", newline="") as file:
       # writer = csv.writer(file)
     #   writer.writerow(["AFTER"])
     #   for node in fiber.nodes:
       #     writer.writerow(node.tolist())

def save_video():
    output_filename = "integrin_simulation.mp4"
    print(f"Saving video to {output_filename}...")
    imageio.mimsave(output_filename, frame_images, fps=1)

QtCore.QTimer.singleShot(1000, frame_1)
QtCore.QTimer.singleShot(2500, frame_2)
QtCore.QTimer.singleShot(4000, frame_3)

app.exec_()
