import numpy as np
import csv
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import imageio
import random

# -------------------- Frame capture --------------------

frame_images = []
frame_count = 0

def capture_frame():
    global frame_count
    img = view.readQImage()
    img = img.convertToFormat(4)
    width, height = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    frame_images.append(arr)
    print(f"Captured frame {frame_count}")
    frame_count += 1

# -------------------- Geometry helpers --------------------

def angle_between(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# -------------------- Integrin + Manager --------------------

class Integrin:
    def __init__(self, view, id, base_position, length=0.004, radius=0.0005, colors=[(1, 0, 0, 1), (0, 0, 1, 1)]):
        self.view = view
        self.id = id
        self.base_position = base_position
        self.length = length
        self.radius = radius
        self.colors = colors
        self.state = 'inactive'
        self.items = []
        self.inactive_length = length
        self.active_length = 2 * length

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
            self.length = self.active_length
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

    def activate_near_fibers(self, fibers, threshold=None, save_path="integrin_distances2.csv"):
        any_activated = False
        data = []
        for integrin in self.integrins.values():
            tip = integrin.get_tip_position()
            min_dist = float('inf')
            closest_pt = None
            for fiber in fibers:
                idx, pt = fiber.get_closest_point(tip)
                dist = np.linalg.norm(tip - pt)
                if dist < min_dist:
                    min_dist = dist
                    closest_pt = pt

            # Use integrin’s *active* length as reach if threshold not provided
            reach = (integrin.length * 2) if threshold is None else threshold

            activated = False
            if integrin.state == 'inactive' and min_dist < reach:
                integrin.switch_to_active()
                activated = True
                any_activated = True

            data.append([integrin.id, *tip, min_dist, activated])

        if save_path is not None:
            with open(save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['ID', 'Tip_X', 'Tip_Y', 'Tip_Z', 'Distance_to_Fiber', 'Activated'])
                writer.writerows(data)
        return any_activated

        

# -------------------- Elastic Fiber FEM --------------------

class ElasticFiberFEM:
    def __init__(self, start_point, end_point, num_points=50, rest_length=0.024, axial_k=2e4, bending_k=1e3, mass=0.1, dt=0.005,
                 descent_iters=50, descent_alpha=1e-8, velocity_damping=0.3):
        self.N = num_points
        self.rest_length = rest_length
        self.axial_k = axial_k
        self.bending_k = bending_k
        self.mass = mass
        self.dt = dt
        self.descent_iters = descent_iters
        self.descent_alpha = descent_alpha
        self.velocity_damping = velocity_damping

        self.start_pos = start_point
        self.end_pos = end_point
        self.x = np.linspace(self.start_pos, self.end_pos, num_points).astype(np.float64)
        self.v = np.zeros_like(self.x)
        self.f_ext = np.zeros_like(self.x)

        self.line = gl.GLLinePlotItem(pos=self.x, color=(1,1,1,1), width=2, antialias=True)
        view.addItem(self.line)

    def update(self):
        self.quasi_static_step()
        self.line.setData(pos=self.x)

    def quasi_static_step(self):
        x_new = np.copy(self.x)
        x_new[0] = self.start_pos
        x_new[-1] = self.end_pos

        for _ in range(self.descent_iters):
            grad = np.zeros_like(x_new)
            for i in range(self.N - 1):
                delta = x_new[i+1] - x_new[i]
                dist = np.linalg.norm(delta)
                if dist < 1e-8:
                    continue
                diff = dist - self.rest_length
                dgrad = self.axial_k * diff * (delta / dist)
                if 0 < i < self.N - 1:
                    grad[i] -= dgrad
                if 0 <= i + 1 < self.N - 1:
                    grad[i + 1] += dgrad

            for i in range(1, self.N - 1):
                b = x_new[i+1] - 2 * x_new[i] + x_new[i-1]
                grad[i-1] += self.bending_k * b
                grad[i]   -= 2 * self.bending_k * b
                grad[i+1] += self.bending_k * b

            grad[1:-1] -= self.f_ext[1:-1]
            for i in range(1, self.N - 1):
                x_new[i] -= self.descent_alpha * grad[i]
                x_new[i] = (1 - self.velocity_damping) * x_new[i] + self.velocity_damping * self.x[i]

            x_new[0] = self.start_pos
            x_new[-1] = self.end_pos

        self.x = x_new
        self.f_ext[:] = 0.0
        self.v[:] = 0.0

    def apply_external_force(self, index, force):
        if 0 < index < self.N - 1:
            self.f_ext[index] += force

    def get_closest_point(self, pos):
        pos = np.array(pos)
        min_dist = float('inf')
        closest_point = None
        closest_idx = None
        for i in range(self.N - 1):
            p1, p2 = self.x[i], self.x[i+1]
            seg = p2 - p1
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-8:
                continue
            unit = seg / seg_len
            proj = np.dot(pos - p1, unit)
            proj = np.clip(proj, 0, seg_len)
            candidate = p1 + proj * unit
            dist = np.linalg.norm(candidate - pos)
            if dist < min_dist:
                min_dist = dist
                closest_point = candidate
                closest_idx = i
        return closest_idx, closest_point

# -------------------- PyQt + Simulation --------------------

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.setWindowTitle('Integrin Activation and Multiple Elastic Fibers')
view.setGeometry(0, 110, 1280, 800)
view.show()
view.setCameraPosition(distance=2.0, elevation=20, azimuth=45)

manager = IntegrinManager(view)
manager.populate_integrins(N=100, radius=0.1, z=0)

fibers = []
for _ in range(10):
    p1 = np.random.uniform(low=[-0.1, -0.1, 0.008], high=[0.1, 0.1, 0.020])
    p2 = np.random.uniform(low=[-0.1, -0.1, 0.008], high=[0.1, 0.1, 0.020])
    fibers.append(ElasticFiberFEM(start_point=p1, end_point=p2))

TOTAL_STEPS = 5          # how many times to re-run "recompute + pull + deform"
STEP_INTERVAL_MS = 500    # ms between steps (visual)
FORCE_GAIN = 1e7       # scale the pull force each step (tune for stability)
REACTIVATE = True         # if True, allow integrins to (re)activate as fibers move

current_step = 0
step_timer = QtCore.QTimer()


def frame_1():
    print("Frame 1: Waiting...")
    capture_frame()

def frame_2():
    print("Frame 2: Drawing fibers and activating integrins...")
    for f in fibers:
        f.line.setData(pos=f.x)
    manager.activate_near_fibers(fibers, threshold=0.01, save_path="integrin_distances2.csv")
    capture_frame()

def frame_3():
    # initialize the CSV headers once
    with open("fiber_deformation2.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Fiber", "Node", "X", "Y", "Z"])

    print("Starting multi-step simulation...")
    step_timer.setInterval(STEP_INTERVAL_MS)
    step_timer.timeout.connect(simulation_step)
    step_timer.start()

def simulation_step():
    global current_step
    try:
        if REACTIVATE:
            manager.activate_near_fibers(fibers, threshold=0.01, save_path=None)

        active_count = sum(1 for i in manager.integrins.values() if i.state == 'active')
        print(f"Step {current_step}: active integrins = {active_count}")

        with open("fiber_deformation2.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            for j, f in enumerate(fibers):
                for i, node in enumerate(f.x):
                    writer.writerow([f"STEP_{current_step}_BEFORE", j, i, *node])

        active_tips = [i.get_tip_position() for i in manager.integrins.values() if i.state == 'active']
        for tip in active_tips:
            min_dist = float('inf')
            best_idx = -1
            best_fiber = None
            best_pt = None
            for f in fibers:
                idx, pt = f.get_closest_point(tip)
                dist = np.linalg.norm(tip - pt)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
                    best_fiber = f
                    best_pt = pt
            if best_fiber is not None and best_pt is not None:
                vec = tip - best_pt
                if np.linalg.norm(vec) > 0:
                    best_fiber.apply_external_force(best_idx, vec * FORCE_GAIN)

        for f in fibers:
            f.update()

        with open("fiber_deformation2.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            for j, f in enumerate(fibers):
                for i, node in enumerate(f.x):
                    writer.writerow([f"STEP_{current_step}_AFTER", j, i, *node])

        capture_frame()

        current_step += 1
        if current_step >= TOTAL_STEPS:
            step_timer.stop()
            print("Simulation loop complete.")
            save_video()

    except Exception as e:
        print("ERROR in simulation_step:", repr(e))
        step_timer.stop()



def save_video():
    print("Saving video to integrin_simulation2.mp4...")
    imageio.mimsave("integrin_simulation2.mp4", frame_images, fps=2)

QtCore.QTimer.singleShot(1000, frame_1)
QtCore.QTimer.singleShot(2500, frame_2)
QtCore.QTimer.singleShot(4000, frame_3)

app.exec_()
