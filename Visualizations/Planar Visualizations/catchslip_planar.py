import numpy as np
import csv
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import imageio
import random

# ==================== Config: Spring + Catch–Slip ====================

# Mechanics: spring force actually applied to fibers (top-segment only)
SPRING_K_SIM = 1e7       # N/m
L0_SHRINK_RATE = 1e-5     # m/s; >0 builds tension over time

# Biophysics for off-rate ONLY: use pN/nm-scale "physical" spring
K_PN_PER_NM = 1.5e-4      # pN per nm effective stiffness for k_off(F)

# Catch–slip (Kong et al. 2009) — parameters in pN converted to N here
# k_off(F) = a1*exp(-((F-b1)/c1)**2) + a2*exp(-((F-b2)/c2)**2)
CATCHSLIP_A1 = 2.2          # 1/s
CATCHSLIP_A2 = 1.2          # 1/s
CATCHSLIP_B1 = 29.9e-12     # N  (29.9 pN)
CATCHSLIP_B2 = 16.2e-12     # N  (16.2 pN)
CATCHSLIP_C1 = 8.4e-12      # N  (8.4 pN)
CATCHSLIP_C2 = 37.8e-12     # N  (37.8 pN)

UNBIND_BEFORE_FORCE = True  # sample unbinding before pushing the fiber this step

# ==================== Frame capture ====================

frame_images = []
frame_count = 0

def capture_frame():
    global frame_count
    img = view.readQImage()
    img = img.convertToFormat(4)  # RGBA8888
    width, height = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    frame_images.append(arr)
    print(f"Captured frame {frame_count}")
    frame_count += 1

# ==================== Bond logging ====================

BOND_EVENTS_CSV = "bond_events.csv"
BOND_SERIES_CSV = "bond_timeseries.csv"
sim_time = 0.0  # seconds, advanced each step

def log_bond_event(event, step, time_s, integ_id, fiber_idx, node_idx,
                   L, L0, e, F_sim, F_phys, koff, lifetime=None):
    # NOTE: L/L0/e here refer to TOP SPRING (pivot->node) quantities
    with open(BOND_EVENTS_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([event, step, f"{time_s:.3f}", integ_id, fiber_idx, node_idx,
                    f"{L:.6e}", f"{L0:.6e}", f"{e:.6e}",
                    f"{F_sim:.6e}", f"{F_phys:.6e}", f"{koff:.6e}",
                    f"{(lifetime if lifetime is not None else 0.0):.3f}"])

def log_bond_state(step, time_s, integ_id, fiber_idx, node_idx,
                   L, L0, e, F_sim, F_phys, koff):
    # NOTE: L/L0/e here refer to TOP SPRING (pivot->node) quantities
    with open(BOND_SERIES_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([step, f"{time_s:.3f}", integ_id, fiber_idx, node_idx,
                    f"{L:.6e}", f"{L0:.6e}", f"{e:.6e}",
                    f"{F_sim:.6e}", f"{F_phys:.6e}", f"{koff:.6e}"])

# ==================== Integrin + Manager ====================

class Integrin:
    def __init__(self, view, id, base_position, length=0.004, radius=0.0005,
                 colors=[(1, 0, 0, 1), (0, 0, 1, 1)]):
        self.view = view
        self.id = id
        self.base_position = np.array(base_position, dtype=np.float64)  # fixed base (z=0)
        self.length = float(length)      # current total length proxy (used for tip calc when inactive)
        self.radius = float(radius)
        self.colors = colors

        self.state = 'inactive'
        self.items = []

        self.inactive_length = float(length)       # full integrin nominal length
        self.inactive_top_length = self.inactive_length * 0.5  # top half nominal
        self.active_length   = 2.0 * float(length)

        # attachment
        self.attachment = None  # (fiber, node_idx)

        # spring params (TOP segment only)
        self.k_spring = SPRING_K_SIM
        self.L0 = self.inactive_top_length        # top spring rest length
        self.L0_shrink_rate = L0_SHRINK_RATE

        # diagnostics
        self.bound_time = 0.0

        # draw initial L-shape
        self.add_inactive_integrin()

    # ---- drawing helpers ----
    def _add_item(self, item, translate=None, rotate=None, scale=None):
        if rotate is not None:
            angle_deg, ax, ay, az = rotate
            item.rotate(angle_deg, ax, ay, az)
        if scale is not None:
            item.scale(*scale)
        if translate is not None:
            item.translate(*translate)
        self.view.addItem(item)
        self.items.append(item)

    def remove_integrin(self):
        for it in self.items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.items = []

    def add_inactive_integrin(self):
        """Two half-stalks + horizontal heads (classic L-shape)."""
        self.remove_integrin()
        x, y, z = self.base_position
        # bottom vertical half-stalks (rigid)
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(cyl,
                           translate=(x + (i - 0.5) * self.radius * 3.0, y, z),
                           scale=(self.radius, self.radius, self.inactive_length / 2.0))
        # top horizontal heads at pivot height
        for i, color in enumerate(self.colors):
            top = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(top,
                           translate=(x + (i - 0.5) * self.radius * 3.0, y, z + self.inactive_length / 2.0),
                           rotate=(90, 1, 0, 0),
                           scale=(self.radius, self.radius, self.inactive_length / 2.0))

    def _pivot_point(self):
        # Hinge between bottom and top pairs
        return self.base_position + np.array([0.0, 0.0, self.inactive_length * 0.5], dtype=np.float64)

    def _draw_active_hinged_to(self, target_pt):
        """
        Hinged draw:
          - Bottom pair: fixed, vertical half-stalks of length inactive_length/2 (no rotation).
          - Top pair: rotate/extend from the pivot at z = base + inactive_length/2 toward target_pt.
        Visual only; mechanics use pivot->target as the spring.
        """
        self.remove_integrin()

        base = self.base_position
        pivot = self._pivot_point()

        # top segment geometry
        to_target = np.array(target_pt, dtype=np.float64) - pivot
        L_top = float(np.linalg.norm(to_target))
        if L_top < 1e-9:
            angle_deg = 0.0
            axis = np.array([1.0, 0.0, 0.0])
        else:
            z_axis = np.array([0.0, 0.0, 1.0])
            u = to_target / max(L_top, 1e-30)
            c = float(np.clip(np.dot(z_axis, u), -1.0, 1.0))
            angle_deg = np.degrees(np.arccos(c))
            axis = np.cross(z_axis, u)
            n = np.linalg.norm(axis)
            axis = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else axis / n

        # α/β lateral offsets (XY only)
        offsets = [np.array([-0.5 * self.radius * 3.0, 0.0, 0.0]),
                   np.array([+0.5 * self.radius * 3.0, 0.0, 0.0])]

        # ---- Bottom pair: fixed vertical half-stalks (do NOT rotate) ----
        x, y, z = base
        L_bottom = self.inactive_length * 0.5
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                cyl,
                translate=(float(x + offsets[i][0]), float(y + offsets[i][1]), float(z)),
                scale=(self.radius, self.radius, L_bottom)
            )

        # ---- Top pair: rotate from the pivot toward the target ----
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                cyl,
                rotate=(angle_deg, float(axis[0]), float(axis[1]), float(axis[2])),
                scale=(self.radius, self.radius, L_top),
                translate=(float(pivot[0] + offsets[i][0]), float(pivot[1] + offsets[i][1]), float(pivot[2]))
            )

        self.state = 'active'
        # keep self.length as a harmless total-length proxy if needed elsewhere
        self.length = L_bottom + L_top

    # ---- state changes ----
    def switch_to_active(self, attach_pt=None):
        if self.state != 'inactive':
            return
        if attach_pt is not None:
            pivot = self._pivot_point()
            L_bind_top = float(np.linalg.norm(np.array(attach_pt, dtype=np.float64) - pivot))
            # top spring rest length initialized to the bind distance (≥ top half nominal)
            self.L0 = max(self.inactive_top_length, L_bind_top)
            self._draw_active_hinged_to(attach_pt)
        else:
            self.L0 = self.inactive_top_length
            self._draw_active_hinged_to(self._pivot_point() + np.array([0, 0, self.L0]))
        self.state = 'active'
        self.bound_time = 0.0

    def unbind(self):
        self.remove_integrin()
        self.state = 'inactive'
        self.attachment = None
        self.length = self.inactive_length
        self.L0 = self.inactive_top_length
        self.bound_time = 0.0
        self.add_inactive_integrin()

    def get_tip_position(self):
        """
        - If ACTIVE & ATTACHED: return the actual fiber node position (so CSV logs post-activation tip).
        - Else: return the vertical tip from base (legacy/inactive case).
        """
        if self.state == 'active' and self.attachment is not None:
            fiber, idx = self.attachment
            idx = int(np.clip(idx, 0, fiber.N - 1))
            return fiber.x[idx].copy()
        x, y, z = self.base_position
        return np.array([x, y, z + self.length], dtype=np.float64)

    @staticmethod
    def catch_slip_off_rate(F):
        # F in Newtons
        F = float(max(0.0, F))
        t1 = ((F - CATCHSLIP_B1) / (CATCHSLIP_C1 + 1e-30)) ** 2
        t2 = ((F - CATCHSLIP_B2) / (CATCHSLIP_C2 + 1e-30)) ** 2
        return CATCHSLIP_A1 * np.exp(-t1) + CATCHSLIP_A2 * np.exp(-t2)


class IntegrinManager:
    def __init__(self, view):
        self.view = view
        self.integrins = {}

    def populate_integrins(self, N, radius, z=0.0):
        # sunflower layout on plane z
        golden_angle = 2.3999632297
        for i in range(N):
            r = radius * np.sqrt(i / float(N))
            theta = i * golden_angle
            x = float(r * np.cos(theta))
            y = float(r * np.sin(theta))
            integ = Integrin(self.view, f"integrin{i+1}", (x, y, float(z)))
            self.integrins[integ.id] = integ

    def activate_near_fibers(self, fibers, threshold=None, save_path="integrin_distances2.csv"):
        """Bind when tip→fiber distance < activated length (unless 'threshold' given).
           Inserts node at closest point; logs a 'bind' event. L-values are TOP SPRING metrics."""
        global current_step, sim_time
        any_activated = False
        rows = []
        for integ in self.integrins.values():
            # If already active, log real tip (Option B)
            if integ.state == 'active':
                tip_now = integ.get_tip_position()
                rows.append([integ.id, *tip_now, 0.0, False])
                continue

            # Inactive: measure to closest point
            tip_pre = integ.get_tip_position()
            min_dist = float('inf')
            best = None
            for f in fibers:
                idx, pt = f.get_closest_point(tip_pre)
                if pt is None:
                    continue
                d = float(np.linalg.norm(tip_pre - pt))
                if d < min_dist:
                    min_dist = d
                    best = (f, idx, pt)

            reach = integ.active_length if threshold is None else float(threshold)
            activated = False
            if best is not None and min_dist < reach:
                f, idx, pt = best
                attach_idx, attach_pt = f.ensure_attachment_node(pt)
                if attach_idx is not None:
                    integ.switch_to_active(attach_pt)
                    integ.attachment = (f, attach_idx)
                    activated = True
                    any_activated = True

                    # ---- logging: bind event (TOP SPRING) ----
                    fib_idx = fibers.index(f)
                    pivot = integ._pivot_point()
                    L_bind_top = float(np.linalg.norm(attach_pt - pivot))
                    e_bind = max(L_bind_top - integ.L0, 0.0)  # should be 0 at bind
                    F_sim_bind = SPRING_K_SIM * e_bind
                    e_nm = e_bind * 1e9
                    F_phys_bind = (K_PN_PER_NM * e_nm) * 1e-12
                    koff_bind = Integrin.catch_slip_off_rate(F_phys_bind)
                    log_bond_event("bind", current_step, sim_time, integ.id, fib_idx, attach_idx,
                                   L_bind_top, integ.L0, e_bind, F_sim_bind, F_phys_bind, koff_bind, lifetime=0.0)

            # Log AFTER potential activation so Tip_* reflects Option B if it just bound
            tip_now = integ.get_tip_position()
            rows.append([integ.id, *tip_now, min_dist, activated])

        if save_path is not None:
            with open(save_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(['ID', 'Tip_X', 'Tip_Y', 'Tip_Z', 'Distance_to_Fiber', 'Activated'])
                w.writerows(rows)
        return any_activated

# ==================== Elastic Fiber FEM ====================

class ElasticFiberFEM:
    def __init__(self, start_point, end_point, num_points=50,
                 axial_k=2e4, bending_k=1e3, mass=0.1, dt=0.005,
                 descent_iters=50, descent_alpha=1e-8, velocity_damping=0.3):
        self.N = int(num_points)
        self.axial_k = float(axial_k)
        self.bending_k = float(bending_k)
        self.mass = float(mass)
        self.dt = float(dt)
        self.descent_iters = int(descent_iters)
        self.descent_alpha = float(descent_alpha)
        self.velocity_damping = float(velocity_damping)

        self.start_pos = np.array(start_point, dtype=np.float64)
        self.end_pos   = np.array(end_point,   dtype=np.float64)

        self.x = np.linspace(self.start_pos, self.end_pos, self.N).astype(np.float64)
        self.v = np.zeros_like(self.x)
        self.f_ext = np.zeros_like(self.x)

        # per-segment rest lengths (critical for stability)
        self.rest_lengths = np.linalg.norm(self.x[1:] - self.x[:-1], axis=1)

        self.line = gl.GLLinePlotItem(pos=self.x, color=(1,1,1,1), width=2, antialias=True)
        view.addItem(self.line)

    def update(self):
        self.quasi_static_step()
        self.line.setData(pos=self.x)

    def quasi_static_step(self):
        x_new = np.copy(self.x)
        x_new[0]  = self.start_pos
        x_new[-1] = self.end_pos

        for _ in range(self.descent_iters):
            grad = np.zeros_like(x_new)

            # axial (per-segment)
            for i in range(self.N - 1):
                delta = x_new[i+1] - x_new[i]
                dist  = np.linalg.norm(delta)
                if dist < 1e-10:
                    continue
                diff   = dist - float(self.rest_lengths[i])
                dgrad  = self.axial_k * diff * (delta / dist)
                if 0 < i < self.N - 1:
                    grad[i]   -= dgrad
                if 0 <= i + 1 < self.N - 1:
                    grad[i+1] += dgrad

            # bending
            for i in range(1, self.N - 1):
                b = x_new[i+1] - 2.0 * x_new[i] + x_new[i-1]
                grad[i-1] += self.bending_k * b
                grad[i]   -= 2.0 * self.bending_k * b
                grad[i+1] += self.bending_k * b

            # external forces
            grad[1:-1] -= self.f_ext[1:-1]

            # integrate (damped)
            for i in range(1, self.N - 1):
                x_new[i] -= self.descent_alpha * grad[i]
                x_new[i]  = (1 - self.velocity_damping) * x_new[i] + self.velocity_damping * self.x[i]

            x_new[0]  = self.start_pos
            x_new[-1] = self.end_pos

        self.x = x_new
        self.f_ext[:] = 0.0
        self.v[:] = 0.0

    def apply_external_force(self, index, force):
        i = int(index)
        if 0 < i < self.N - 1:
            self.f_ext[i] += np.array(force, dtype=np.float64)

    def get_closest_point(self, pos):
        pos = np.array(pos, dtype=np.float64)
        min_dist = float('inf')
        closest_point = None
        closest_idx = None
        for i in range(self.N - 1):
            p1, p2 = self.x[i], self.x[i+1]
            seg = p2 - p1
            L = np.linalg.norm(seg)
            if L < 1e-12:
                continue
            u = seg / L
            t = np.dot(pos - p1, u)
            t = float(np.clip(t, 0.0, L))
            cand = p1 + t * u
            d = float(np.linalg.norm(cand - pos))
            if d < min_dist:
                min_dist = d
                closest_point = cand
                closest_idx = i
        return closest_idx, closest_point

    def insert_node(self, seg_index, point):
        """Insert a node right after seg_index at exact 'point' on that segment; split rest length."""
        seg_index = int(seg_index)
        point = np.array(point, dtype=np.float64)

        # insert geometry
        self.x     = np.insert(self.x,     seg_index + 1, point, axis=0)
        self.v     = np.insert(self.v,     seg_index + 1, 0.0,   axis=0)
        self.f_ext = np.insert(self.f_ext, seg_index + 1, 0.0,   axis=0)

        # split old rest length proportionally
        p0 = self.x[seg_index]
        p1 = self.x[seg_index + 2]     # end of original segment (shifted after insert)
        seg_vec = p1 - p0
        L = float(np.linalg.norm(seg_vec)) + 1e-12
        u = seg_vec / L
        t = float(np.dot(self.x[seg_index + 1] - p0, u))   # distance along segment to new point
        t = max(0.0, min(L, t))

        old_rl = float(self.rest_lengths[seg_index]) if seg_index < len(self.rest_lengths) else L
        rl_a = old_rl * (t / L)
        rl_b = old_rl * (1.0 - t / L)

        self.rest_lengths = np.insert(self.rest_lengths, seg_index + 1, rl_b, axis=0)
        self.rest_lengths[seg_index] = rl_a

        self.N = self.x.shape[0]
        if self.line is not None:
            self.line.setData(pos=self.x)
        return seg_index + 1

    def ensure_attachment_node(self, pos, min_sep=1e-8):
        idx, pt = self.get_closest_point(pos)
        if idx is None or pt is None:
            return None, None
        if np.linalg.norm(pt - self.x[idx])   <= min_sep: return idx,     self.x[idx]
        if np.linalg.norm(pt - self.x[idx+1]) <= min_sep: return idx + 1, self.x[idx+1]
        new_idx = self.insert_node(idx, pt)
        return new_idx, self.x[new_idx]

# ==================== PyQt + Simulation ====================

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.setWindowTitle('Integrin Springs (Top Only) + Catch–Slip (hinged)')
view.setGeometry(0, 110, 1280, 800)
view.setCameraPosition(distance=2.0, elevation=20, azimuth=45)
view.show()

manager = IntegrinManager(view)
manager.populate_integrins(N=100, radius=0.1, z=0.0)  # plane z=0

fibers = []
for _ in range(10):
    p1 = np.random.uniform(low=[-0.1, -0.1, 0.007], high=[0.1, 0.1, 0.011])
    p2 = np.random.uniform(low=[-0.1, -0.1, 0.007], high=[0.1, 0.1, 0.011])
    fibers.append(ElasticFiberFEM(start_point=p1, end_point=p2))

TOTAL_STEPS = 5
STEP_INTERVAL_MS = 750
REACTIVATE = True

current_step = 0
step_timer = QtCore.QTimer()

def frame_1():
    print("Frame 1: Waiting...")
    capture_frame()

def frame_2():
    print("Frame 2: Set up logs, draw fibers, activate integrins...")
    # Create bond logs BEFORE activation so binds get recorded
    with open(BOND_EVENTS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event","step","time_s","integrin","fiber_idx","node_idx",
                    "L","L0","extension_e","F_sim","F_phys","koff","lifetime_s"])
    with open(BOND_SERIES_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step","time_s","integrin","fiber_idx","node_idx",
                    "L","L0","extension_e","F_sim","F_phys","koff"])

    for f in fibers:
        f.line.setData(pos=f.x)

    manager.activate_near_fibers(fibers, threshold=None, save_path="integrin_distances2.csv")
    capture_frame()

def frame_3():
    # deformation log
    with open("fiber_deformation2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Step", "Fiber", "Node", "X", "Y", "Z"])

    print("Starting multi-step simulation...")
    step_timer.setInterval(STEP_INTERVAL_MS)
    step_timer.timeout.connect(simulation_step)
    step_timer.start()

def simulation_step():
    global current_step, sim_time
    try:
        dt = STEP_INTERVAL_MS / 1000.0

        if REACTIVATE:
            manager.activate_near_fibers(fibers, threshold=None, save_path=None)

        active = [i for i in manager.integrins.values() if i.state == 'active']
        print(f"Step {current_step}: active integrins = {len(active)}")

        # log BEFORE
        with open("fiber_deformation2.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            for j, f in enumerate(fibers):
                for i, node in enumerate(f.x):
                    w.writerow([f"STEP_{current_step}_BEFORE", j, i, *node])

        still_bound = []
        for integ in active:
            if integ.attachment is None:
                continue
            fib, idx = integ.attachment
            attach_pt = fib.x[idx]
            base = integ.base_position
            pivot = integ._pivot_point()

            # active contraction (rest-length change) of TOP spring only, with hard floor
            integ.L0 = max(integ.inactive_top_length, float(integ.L0) - float(integ.L0_shrink_rate) * dt)

            # TOP spring geometry: pivot -> attach
            d_vec_top = attach_pt - pivot
            L_top = float(np.linalg.norm(d_vec_top))
            if L_top < 1e-9:
                # nothing to pull
                integ._draw_active_hinged_to(attach_pt)
                continue

            u_top = d_vec_top / L_top
            # (optional) keep a total length proxy only for visuals/legacy tip calc
            integ.length = (integ.inactive_length * 0.5) + L_top
            integ.bound_time += dt

            # extension and forces (TOP spring)
            e = max(L_top - integ.L0, 0.0)    # no compression; cannot pull fiber below pivot+top-rest
            F_sim  = SPRING_K_SIM * e
            e_nm   = e * 1e9
            F_phys = (K_PN_PER_NM * e_nm) * 1e-12
            koff   = Integrin.catch_slip_off_rate(F_phys)

            # per-step state log (TOP metrics)
            fib_idx = fibers.index(fib)
            log_bond_state(current_step, sim_time, integ.id, fib_idx, idx,
                           L_top, integ.L0, e, F_sim, F_phys, koff)

            # stochastic unbinding BEFORE applying fiber force
            p_unbind = 1.0 - np.exp(-koff * dt)
            if UNBIND_BEFORE_FORCE and random.random() < p_unbind:
                log_bond_event("unbind", current_step, sim_time, integ.id, fib_idx, idx,
                               L_top, integ.L0, e, F_sim, F_phys, koff, lifetime=integ.bound_time)
                integ.unbind()
                continue

            # draw (hinged) and apply mechanical force from TOP spring
            integ.remove_integrin()
            integ._draw_active_hinged_to(attach_pt)
            if F_sim > 0:
                fib.apply_external_force(idx, -F_sim * u_top)

            still_bound.append((integ, fib, idx))

        # update fibers
        for f in fibers:
            f.update()

        # redraw after update (and optionally unbind after force)
        for integ, fib, idx in still_bound:
            attach_post = fib.x[idx]
            pivot = integ._pivot_point()

            d_top_post = attach_post - pivot
            L_top_post = float(np.linalg.norm(d_top_post))
            # keep the display in sync
            integ.remove_integrin()
            integ._draw_active_hinged_to(attach_post)

            if not UNBIND_BEFORE_FORCE:
                e_post = max(L_top_post - integ.L0, 0.0)
                F_sim_post  = SPRING_K_SIM * e_post
                F_phys_post = (K_PN_PER_NM * (e_post * 1e9)) * 1e-12
                koff_post   = Integrin.catch_slip_off_rate(F_phys_post)
                p_unbind = 1.0 - np.exp(-koff_post * dt)
                if random.random() < p_unbind:
                    fib_idx = fibers.index(fib)
                    log_bond_event("unbind", current_step, sim_time, integ.id, fib_idx, idx,
                                   L_top_post, integ.L0, e_post, F_sim_post, F_phys_post, koff_post,
                                   lifetime=integ.bound_time)
                    integ.unbind()
                    continue

        # log AFTER
        with open("fiber_deformation2.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            for j, f in enumerate(fibers):
                for i, node in enumerate(f.x):
                    w.writerow([f"STEP_{current_step}_AFTER", j, i, *node])

        capture_frame()

        # advance time/step
        dt = STEP_INTERVAL_MS / 1000.0
        sim_time += dt
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
