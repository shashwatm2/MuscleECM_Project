import numpy as np
import csv
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore, QtGui
import imageio
import random
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # write PNGs without opening windows
import matplotlib.pyplot as plt
import pandas as pd

# ==================== Overlay labels (2D QLabel projected from 3D) ====================

LABEL_FONT_PT    = 10.0
LABEL_COLOR_CSS  = "rgba(255,255,0,0.98)"  # bright yellow
LABEL_PAD        = 1

class OverlayLabeler(QtCore.QObject):
    """Projects 3D points to 2D and shows tiny QLabel numbers over the GL view."""
    def __init__(self, view: gl.GLViewWidget):
        super().__init__(parent=view)
        self.view = view
        self.labels = {}     # key -> QLabel
        self.world_pos = {}  # key -> np.array([x,y,z])
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self.refresh_all)
        self.timer.start()
        view.resizeEvent = self._wrap_resize(view.resizeEvent)

    def _wrap_resize(self, old_handler):
        def new(ev):
            if old_handler is not None:
                old_handler(ev)
            self.refresh_all()
        return new

    def set_label(self, key: str, text: str, world_pos):
        lbl = self.labels.get(key)
        if lbl is None:
            lbl = QtWidgets.QLabel(self.view)
            lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            lbl.setStyleSheet(
                f"color:{LABEL_COLOR_CSS};"
                "background:transparent;"
                f"padding:{LABEL_PAD}px;"
                "font-family:Arial, Helvetica, sans-serif;"
                f"font-size:{LABEL_FONT_PT}pt;"
            )
            self.labels[key] = lbl
        if lbl.text() != text:
            lbl.setText(text)
            lbl.adjustSize()
        self.world_pos[key] = np.array(world_pos, dtype=float)
        self._move_one(key)

    def refresh_all(self):
        for k in list(self.world_pos.keys()):
            self._move_one(k)

    def _move_one(self, key):
        pos = self.world_pos.get(key); lbl = self.labels.get(key)
        if pos is None or lbl is None: return
        proj = self.view.projectionMatrix()
        viewm = self.view.viewMatrix()
        mvp = QtGui.QMatrix4x4(proj); mvp *= viewm
        v4 = QtGui.QVector4D(float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
        p4 = mvp.map(v4); w = p4.w()
        if w == 0: lbl.hide(); return
        x_ndc, y_ndc, z_ndc = p4.x()/w, p4.y()/w, p4.z()/w
        if z_ndc < -1.0 or z_ndc > 1.0 or abs(x_ndc) > 1.2 or abs(y_ndc) > 1.2:
            lbl.hide(); return
        vw, vh = max(1, self.view.width()), max(1, self.view.height())
        x_px = int((x_ndc * 0.5 + 0.5) * vw - lbl.width()/2)
        y_px = int(((1.0 - (y_ndc * 0.5 + 0.5)) * vh) - lbl.height()/2)
        lbl.move(x_px, y_px); lbl.show()

# ==================== Config: Spring + Catch–Slip (integrins) ====================

SPRING_K_SIM   = 5.5e6      # N/m (top segment only)
K_PN_PER_NM    = 2e-5       # pN per nm for off-rate model

# preload: constant force at first attachment
BASE_PRELOAD_PN = 200.0     # pN, you can change this

# Catch–slip (Kong et al. 2009)
CATCHSLIP_A1 = 1.8          # 1/s
CATCHSLIP_A2 = 1.2          # 1/s
CATCHSLIP_B1 = 2.774e-10    # N
CATCHSLIP_B2 = 4.0e-11      # N
CATCHSLIP_C1 = 4.95e-11     # N
CATCHSLIP_C2 = 3.0e-11      # N

UNBIND_BEFORE_FORCE = True

# ==================== Config: Fiber↔Fiber crosslinks ====================

KFF_ON   = 0.01         # s^-1
KFF_OFF0 = 1.0e-4       # s^-1
BELL_DX  = 0.5e-9       # m
TEMP_K   = 310.0        # K
KB       = 1.380649e-23 # J/K
KB_T     = KB * TEMP_K  # J
EF_COLLAGEN = 1.1e6     # Pa  (1.1 MPa)
DF_COLLAGEN = 180e-9    # m   (diameter = 180 nm)

CROSSLINK_K_SIM  = 1e6
CROSSLINK_RANGE  = 0.025
CROSSLINK_STRIDE = 1
DRAW_CROSSLINKS  = True

IONIC_FRACTION = 0.5
IONIC_COLOR    = (1, 0, 0, 1)   # red
COVALENT_COLOR = (0, 1, 0, 1)   # green

IONIC_K_MULT        = 0.6
IONIC_KOFF0_MULT    = 5.0
IONIC_DX_MULT       = 1.5
COVALENT_K_MULT     = 1.4
COVALENT_KOFF0_MULT = 0.0
COVALENT_DX_MULT    = 0.7

KFF_FORCE_CLAMP_PN = 500.0
KFF_FORCE_CLAMP_N  = KFF_FORCE_CLAMP_PN * 1e-12
KFF_EXP_CLAMP      = 60.0
KFF_PN_PER_NM      = 1e-3

# ==================== Cell placement & scale ====================

CELL_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
CELL_RADIUS = 0.04

NUM_INTEGRINS = 10
ADD_NEW_CELL_EACH_STEP = True

# ==================== Voronoi input ====================

VOR_VERTICES_PATH = "test-file_vertices.out"
VOR_EDGES_PATH    = "test-file_nodes_to_edges.out"
VORONOI_SCALE     = 1
CENTER_VORONOI    = True
FIBER_LIMIT       = 100

# ==================== CSV paths ====================

INTEGRIN_DIST_CSV = "integrin_distances3.csv"

# ==================== Frame capture ====================

frame_images = []
frame_count = 0
sim_time = 0.0  # seconds

def capture_frame():
    global frame_count
    img = view.readQImage().convertToFormat(4)  # RGBA8888
    width, height = img.width(), img.height()
    ptr = img.bits(); ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    frame_images.append(arr)
    print(f"Captured frame {frame_count}")
    frame_count += 1

# ==================== Logging ====================

BOND_EVENTS_CSV = "bond_events2.csv"
BOND_SERIES_CSV = "bond_timeseries2.csv"
XL_EVENTS_CSV   = "crosslink_events2.csv"
XL_SERIES_CSV   = "crosslink_timeseries2.csv"
INTEGRIN_FORCE_CSV = "integrin_forces.csv"

def log_bond_event(event, step, time_s, integ_id, fiber_idx, node_idx,
                   L, L0, e, F_sim, F_phys, koff, lifetime=None):
    with open(BOND_EVENTS_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([event, step, f"{time_s:.3f}", integ_id, fiber_idx, node_idx,
                    f"{L:.6e}", f"{L0:.6e}", f"{e:.6e}",
                    f"{F_sim:.6e}", f"{F_phys:.6e}", f"{koff:.6e}",
                    f"{(lifetime if lifetime is not None else 0.0):.3f}"])

def log_bond_state(step, time_s, integ_id, fiber_idx, node_idx,
                   L, L0, e, F_sim, F_phys, koff):
    with open(BOND_SERIES_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([step, f"{time_s:.3f}", integ_id, fiber_idx, node_idx,
                    f"{L:.6e}", f"{L0:.6e}", f"{e:.6e}",
                    f"{F_sim:.6e}", f"{F_phys:.6e}", f"{koff:.6e}"])

def log_xl_event(event, step, time_s, fi, ni, fj, nj, L, L0, F, koff):
    with open(XL_EVENTS_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([event, step, f"{time_s:.3f}", fi, ni, fj, nj,
                    f"{L:.6e}", f"{L0:.6e}", f"{F:.6e}", f"{koff:.6e}"])

def log_xl_state(step, time_s, fi, ni, fj, nj, L, L0, F, koff):
    with open(XL_SERIES_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([step, f"{time_s:.3f}", fi, ni, fj, nj,
                    f"{L:.6e}", f"{L0:.6e}", f"{F:.6e}", f"{koff:.6e}"])
        
def log_integrin_force(step, time_s, integ_id, fiber_idx, node_idx, e, F_sim, F_phys):
    with open(INTEGRIN_FORCE_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([step, f"{time_s:.3f}", integ_id, fiber_idx, node_idx,
                    f"{e:.6e}", f"{F_sim:.6e}", f"{F_phys:.6e}"])

# ==================== Helpers ====================

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _axis_angle_from_z(to_vec):
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = _unit(to_vec)
    c = float(np.clip(np.dot(z, u), -1.0, 1.0))
    angle = np.degrees(np.arccos(c))
    axis = np.cross(z, u)
    n = np.linalg.norm(axis)
    axis = np.array([1.0, 0.0, 0.0]) if n < 1e-12 else axis / n
    return angle, float(axis[0]), float(axis[1]), float(axis[2])

def fibonacci_sphere_points(n, radius, center):
    phi = (1 + np.sqrt(5)) / 2.0
    pts = []
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1 - y*y))
        theta = 2 * np.pi * i / phi
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        pts.append(center + radius * np.array([x, y, z], dtype=np.float64))
    return pts

# ==================== Integrin + Manager ====================

class Integrin:
    def __init__(self, view, id, base_position, length=0.03, radius=0.0025,
                 colors=[(1, 0, 0, 1), (0, 0, 1, 1)], cell_center=CELL_CENTER,
                 label_num=None):
        self.view = view
        self.id = id
        self.base_position = np.array(base_position, dtype=np.float64)
        self.center = np.array(cell_center, dtype=np.float64)
        self.length = float(length)
        self.radius = float(radius)
        self.colors = colors

        self.state = 'inactive'
        self.items = []

        self.inactive_length = float(length)
        self.inactive_top_length = self.inactive_length * 0.5
        self.active_length   = 2.0 * float(length)
        self.top_max = self.inactive_length

        self.attachment = None  # (fiber, node_idx)
        self.k_spring = SPRING_K_SIM
        self.L0 = self.inactive_top_length
        self.bound_time = 0.0

        self.n = _unit(self.base_position - self.center)
        ref = np.array([0.0, 0.0, 1.0]) if abs(self.n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        self.t1 = _unit(np.cross(ref, self.n))
        self.t2 = _unit(np.cross(self.n, self.t1))

        self.label_num = label_num  # integer
        self.add_inactive_integrin()

    def _label_pos(self):
        return self.base_position + self.n * (self.inactive_length * 1.2)

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

    def _pivot_point(self):
        return self.base_position + self.n * self.inactive_length

    def add_inactive_integrin(self):
        self.remove_integrin()
        base = self.base_position
        pivot = self._pivot_point()

        rot_to_n = _axis_angle_from_z(self.n)
        rot_to_tangent = _axis_angle_from_z(self.t2)

        offsets = [-0.5 * self.radius * 3.0 * self.t1,
                   +0.5 * self.radius * 3.0 * self.t1]

        # bottom stalks
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                cyl,
                translate=tuple((base + offsets[i]).astype(float)),
                rotate=rot_to_n,
                scale=(self.radius, self.radius, self.inactive_length)
            )

        # top horizontal heads
        for i, color in enumerate(self.colors):
            top = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                top,
                translate=tuple((pivot + offsets[i]).astype(float)),
                rotate=rot_to_tangent,
                scale=(self.radius, self.radius, self.inactive_length)
            )

        if self.label_num is not None:
            overlay.set_label(f"integ:{self.label_num}", str(self.label_num), self._label_pos())

    def _draw_active_hinged_to(self, target_pt):
        self.remove_integrin()
        base = self.base_position
        pivot = self._pivot_point()

        to_target = np.array(target_pt, dtype=np.float64) - pivot
        L_top_raw = float(np.linalg.norm(to_target))
        if L_top_raw < 1e-9:
            angle_deg = 0.0
            axis = np.array([1.0, 0.0, 0.0])
            L_top = 0.0
        else:
            u = to_target / L_top_raw
            L_top = min(L_top_raw, self.top_max)
            z_axis = np.array([0.0, 0.0, 1.0])
            c = float(np.clip(np.dot(z_axis, u), -1.0, 1.0))
            angle_deg = np.degrees(np.arccos(c))
            axis = np.cross(z_axis, u)
            n = np.linalg.norm(axis)
            axis = np.array([1.0, 0.0, 0.0]) if n < 1e-12 else axis / n

        offsets = [-0.5 * self.radius * 3.0 * self.t1,
                   +0.5 * self.radius * 3.0 * self.t1]

        rot_to_n = _axis_angle_from_z(self.n)
        L_bottom = self.inactive_length
        # bottom stalks
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                cyl,
                translate=tuple((base + offsets[i]).astype(float)),
                rotate=rot_to_n,
                scale=(self.radius, self.radius, L_bottom)
            )

        # top hinged pair
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                cyl,
                translate=tuple((pivot + offsets[i]).astype(float)),
                rotate=(angle_deg, float(axis[0]), float(axis[1]), float(axis[2])),
                scale=(self.radius, self.radius, L_top)
            )

        self.state = 'active'
        self.length = L_bottom + L_top

        if self.label_num is not None:
            overlay.set_label(f"integ:{self.label_num}", str(self.label_num), self._label_pos())

    def switch_to_active(self, attach_pt=None):
        """
        Activate integrin. If attach_pt is provided, hinge the top segment to that point.
        L0 is assumed already set by caller (for preload) and is not modified here.
        """
        if self.state != 'inactive':
            return
        if attach_pt is not None:
            pivot = self._pivot_point()
            if attach_pt[2] < pivot[2]:
                return
            L_bind_top = float(np.linalg.norm(np.array(attach_pt, dtype=np.float64) - pivot))
            if L_bind_top > self.top_max:
                return
            # DO NOT change self.L0 here; caller already set it (possibly with preload).
            self._draw_active_hinged_to(attach_pt)
        else:
            # fallback: no attachment point, use geometric default
            self.L0 = self.inactive_top_length
            self._draw_active_hinged_to(self._pivot_point() + self.n * self.L0)

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
        if self.state == 'active' and self.attachment is not None:
            fiber, idx = self.attachment
            idx = int(np.clip(idx, 0, fiber.N - 1))
            return fiber.x[idx].copy()
        return self.base_position + self.n * self.length

    @staticmethod
    def catch_slip_off_rate(F):
        F = float(max(0.0, F))
        t1 = ((F - CATCHSLIP_B1) / (CATCHSLIP_C1 + 1e-30)) ** 2
        t2 = ((F - CATCHSLIP_B2) / (CATCHSLIP_C2 + 1e-30)) ** 2
        return CATCHSLIP_A1 * np.exp(-t1) + CATCHSLIP_A2 * np.exp(-t2)

class IntegrinManager:
    def __init__(self, view):
        self.view = view
        self.integrins = {}
        self.counter = 0

    def populate_integrins_on_sphere(self, N, center, radius):
        base_pts = fibonacci_sphere_points(N, radius, center)
        for p in base_pts:
            self.counter += 1
            integ = Integrin(self.view, f"integrin{self.counter}", p, cell_center=center, label_num=self.counter)
            self.integrins[integ.id] = integ

    def activate_near_fibers(self, fibers, threshold=None, save_path=None):
        global current_step, sim_time
        any_activated = False
        rows = []
        for integ in self.integrins.values():
            if integ.state == 'active':
                tip_now = integ.get_tip_position()
                rows.append([integ.id, *tip_now, 0.0, False])
                continue

            tip_pre = integ.get_tip_position()
            min_dist = float('inf'); best = None
            for f in fibers:
                idx, pt = f.get_closest_point(tip_pre)
                if pt is None: continue
                d = float(np.linalg.norm(tip_pre - pt))
                if d < min_dist:
                    min_dist = d; best = (f, idx, pt)

            reach = integ.active_length if threshold is None else float(threshold)
            activated = False
            if best is not None and min_dist < reach:
                f, idx, pt = best
                attach_idx, attach_pt = f.ensure_attachment_node(pt)
                if attach_idx is not None:
                    # Z-gate here as well
                    pivot = integ._pivot_point()
                    if attach_pt[2] >= pivot[2]:
                        L_bind_top = float(np.linalg.norm(attach_pt - pivot))
                        if L_bind_top <= integ.top_max:
                            # --- compute preload extension in meters using L_bind_top + e0 ---
                            e0_nm = BASE_PRELOAD_PN / K_PN_PER_NM      # nm
                            e0 = e0_nm * 1e-9                           # m
                            # Your requested sign: L0 = L_bind_top + e0
                            integ.L0 = L_bind_top + e0

                            integ.switch_to_active(attach_pt)
                            integ.attachment = (f, attach_idx)
                            activated = True; any_activated = True

                            fib_idx = fibers.index(f) + 1  # 1-indexed for logs
                            # At bind, L_top = L_bind_top
                            e_bind = abs(L_bind_top - integ.L0)  # ≈ e0
                            F_sim_bind = SPRING_K_SIM * e_bind
                            e_nm = e_bind * 1e9
                            F_phys_bind = (K_PN_PER_NM * e_nm) * 1e-12
                            koff_bind = Integrin.catch_slip_off_rate(F_phys_bind)
                            log_bond_event("bind", current_step, sim_time, integ.id, fib_idx, attach_idx,
                                           L_bind_top, integ.L0, e_bind, F_sim_bind, F_phys_bind, koff_bind, lifetime=0.0)

            tip_now = integ.get_tip_position()
            rows.append([integ.id, *tip_now, (min_dist if np.isfinite(min_dist) else 0.0), activated])

        if save_path is not None:
            header_needed = (not os.path.exists(save_path)) or (os.path.getsize(save_path) == 0)
            with open(save_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if header_needed:
                    w.writerow(['Step','Time_s','ID','Tip_X','Tip_Y','Tip_Z','Distance_to_Fiber','Activated'])
                for r in rows:
                    w.writerow([current_step, f"{sim_time:.3f}", *r])

        return any_activated

# ==================== Elastic Fiber FEM (no endpoint clamp, no padding) ====================

class ElasticFiberFEM:
    def __init__(self, start_point, end_point, num_points=50,
                 axial_k=2e4, bending_k=1e3, mass=0.1, dt=0.005,
                 descent_iters=50, descent_alpha=1.2e-8, velocity_damping=0.8):
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

        self.rest_lengths = np.linalg.norm(self.x[1:] - self.x[:-1], axis=1)

        # store original/rest geometry (kept in sync on node insert)
        self.x_rest = np.linspace(self.start_pos, self.end_pos, self.N).astype(np.float64)

        self.line = gl.GLLinePlotItem(pos=self.x, color=(1,1,1,1), width=2, antialias=True)
        view.addItem(self.line)

        self.index = None  # will be assigned and labeled after creation

    def _midpoint(self):
        return np.mean(self.x, axis=0)

    def update(self):
        self.quasi_static_step()
        self.line.setData(pos=self.x)
        if self.index is not None:
            overlay.set_label(f"fiber:{self.index}", str(self.index), self._midpoint())

    def quasi_static_step(self):
        # Free-endpoint quasi-static relaxation (with endpoint clamping)
        x_new = np.copy(self.x)
        x_new[0]  = self.start_pos
        x_new[-1] = self.end_pos

        for _ in range(self.descent_iters):
            grad = np.zeros_like(x_new)

            # axial (per-segment)
            for i in range(self.N - 1):
                delta = x_new[i+1] - x_new[i]
                dist  = np.linalg.norm(delta)
                if dist < 1e-10: continue
                diff   = dist - float(self.rest_lengths[i])
                dgrad  = self.axial_k * diff * (delta / dist)
                grad[i]   -= dgrad
                grad[i+1] += dgrad

            # bending
            for i in range(1, self.N - 1):
                b = x_new[i+1] - 2.0 * x_new[i] + x_new[i-1]
                grad[i-1] += self.bending_k * b
                grad[i]   -= 2.0 * self.bending_k * b
                grad[i+1] += 2.0 * self.bending_k * b

            # external forces
            grad -= self.f_ext

            # gradient descent + under-relaxation on nodes (not including endpoints)
            for i in range(self.N):
                x_new[i] -= self.descent_alpha * grad[i]
                x_new[i]  = (1 - self.velocity_damping) * x_new[i] + self.velocity_damping * self.x[i]

            x_new[0]  = self.start_pos
            x_new[-1] = self.end_pos
            
            # NOTE: no projection / padding logic here anymore

        self.x = x_new
        self.f_ext[:] = 0.0
        self.v[:] = 0.0

    def apply_external_force(self, index, force):
        i = int(np.clip(index, 0, self.N - 1))
        self.f_ext[i] += np.array(force, dtype=np.float64)

    def get_closest_point(self, pos):
        pos = np.array(pos, dtype=np.float64)
        min_dist = float('inf'); closest_point = None; closest_idx = None
        for i in range(self.N - 1):
            p1, p2 = self.x[i], self.x[i+1]
            seg = p2 - p1
            L = np.linalg.norm(seg)
            if L < 1e-12: continue
            u = seg / L
            t = np.dot(pos - p1, u)
            t = float(np.clip(t, 0.0, L))
            cand = p1 + t * u
            d = float(np.linalg.norm(cand - pos))
            if d < min_dist:
                min_dist = d; closest_point = cand; closest_idx = i
        return closest_idx, closest_point

    def insert_node(self, seg_index, point):
        seg_index = int(seg_index)
        point = np.array(point, dtype=np.float64)

        self.x     = np.insert(self.x,     seg_index + 1, point, axis=0)
        self.v     = np.insert(self.v,     seg_index + 1, 0.0,   axis=0)
        self.f_ext = np.insert(self.f_ext, seg_index + 1, 0.0,   axis=0)

        p0 = self.x[seg_index]
        p1 = self.x[seg_index + 2]
        seg_vec = p1 - p0
        L = float(np.linalg.norm(seg_vec)) + 1e-12
        u = seg_vec / L
        t = float(np.dot(self.x[seg_index + 1] - p0, u))
        t = max(0.0, min(L, t))

        old_rl = float(self.rest_lengths[seg_index]) if seg_index < len(self.rest_lengths) else L
        rl_a = old_rl * (t / L)
        rl_b = old_rl * (1.0 - t / L)
        self.rest_lengths = np.insert(self.rest_lengths, seg_index + 1, rl_b, axis=0)
        self.rest_lengths[seg_index] = rl_a

        r0 = self.x_rest[seg_index]
        r1 = self.x_rest[seg_index + 1]
        frac = (t / L)
        rest_insert = r0 + frac * (r1 - r0)
        self.x_rest = np.insert(self.x_rest, seg_index + 1, rest_insert, axis=0)

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

    def reset_to_rest(self, alpha=1.0):
        if self.x_rest.shape[0] != self.x.shape[0]:
            self.x_rest = np.linspace(self.start_pos, self.end_pos, self.N)
        self.x[:] = (1.0 - alpha) * self.x + alpha * self.x_rest
        self.v[:] = 0.0
        self.f_ext[:] = 0.0

# ==================== Crosslinks ====================

class Crosslink:
    def __init__(self, view, fi, ni, fj, nj, L0, k=CROSSLINK_K_SIM, xtype='ionic'):
        self.fi, self.ni = int(fi), int(ni)
        self.fj, self.nj = int(fj), int(nj)
        self.k = float(k)
        self.L0 = float(L0)
        self.xtype = xtype
        if self.xtype == 'ionic':
            self.k = float(k) * IONIC_K_MULT
            self.koff0 = KFF_OFF0 * IONIC_KOFF0_MULT
            self.dx = BELL_DX * IONIC_DX_MULT
            self.color = IONIC_COLOR
        else:
            self.k = float(k) * COVALENT_K_MULT
            self.koff0 = KFF_OFF0 * COVALENT_KOFF0_MULT
            self.dx = BELL_DX * COVALENT_DX_MULT
            self.color = COVALENT_COLOR
        self.line = None
        if DRAW_CROSSLINKS:
            self.line = gl.GLLinePlotItem(pos=np.zeros((2,3)), color=self.color, width=1, antialias=True)
            view.addItem(self.line)

    def key(self):
        a = (self.fi, self.ni); b = (self.fj, self.nj)
        return tuple(sorted((a, b)))

    def values(self, fibers):
        pi = fibers[self.fi].x[self.ni]
        pj = fibers[self.fj].x[self.nj]
        d  = pj - pi
        L  = float(np.linalg.norm(d))
        u  = d / L if L > 1e-12 else np.array([0.0,0.0,0.0])
        e  = (L - self.L0) if self.xtype == 'covalent' else max(L - self.L0, 0.0)
        F  = self.k * e
        return pi, pj, u, L, e, F

    def apply_force(self, fibers):
        pi, pj, u, L, e, F = self.values(fibers)
        fibers[self.fi].apply_external_force(self.ni, +F * u)
        fibers[self.fj].apply_external_force(self.nj, -F * u)
        return L, F

    def redraw(self, fibers):
        if self.line is None: return
        pi = fibers[self.fi].x[self.ni]
        pj = fibers[self.fj].x[self.nj]
        self.line.setData(pos=np.vstack([pi, pj]), color=self.color)

    def koff(self, F):
        if getattr(self, 'koff0', 0.0) == 0.0:
            return 0.0
        F_eff = min(max(0.0, float(F)), KFF_FORCE_CLAMP_N)
        x = (F_eff * self.dx) / (KB_T + 1e-30)
        x = max(0.0, min(x, KFF_EXP_CLAMP))
        return self.koff0 * np.exp(x)

    def remove_from_view(self, view):
        if self.line is not None:
            try:
                view.removeItem(self.line)
            except Exception:
                pass
            self.line = None

class CrosslinkManager:
    def __init__(self, view):
        self.view = view
        self.links = {}

    def _pair_iter(self, fibers):
        for fi, f in enumerate(fibers):
            for fj in range(fi+1, len(fibers)):
                g = fibers[fj]
                for ni in range(0, f.N, CROSSLINK_STRIDE):
                    for nj in range(0, g.N, CROSSLINK_STRIDE):
                        yield fi, ni, fj, nj, f.x[ni], g.x[nj]

    def formation_step(self, fibers, dt):
        for fi, ni, fj, nj, pi, pj in self._pair_iter(fibers):
            d = float(np.linalg.norm(pj - pi))
            if d > CROSSLINK_RANGE:
                continue
            key = tuple(sorted(((fi,ni),(fj,nj))))
            if key in self.links:
                continue
            p_on = 1.0 - np.exp(-KFF_ON * dt)
            if random.random() < p_on:
                xtype = 'ionic' if random.random() < IONIC_FRACTION else 'covalent'
                xl = Crosslink(self.view, fi, ni, fj, nj, L0=d, k=CROSSLINK_K_SIM, xtype=xtype)
                self.links[xl.key()] = xl
                # 1-indexed in logs
                log_xl_event("bind", current_step, sim_time, fi+1, ni, fj+1, nj, d, d, 0.0, KFF_OFF0)

    def force_and_break_step(self, fibers, dt):
        dead = []
        for key, xl in list(self.links.items()):
            L, F_sim = xl.apply_force(fibers)
            e_m  = max(L - xl.L0, 0.0)
            e_nm = e_m * 1e9
            F_off_pN = e_nm * KFF_PN_PER_NM
            F_off_N  = F_off_pN * 1e-12
            koff = xl.koff(F_off_N)
            p_off = -np.expm1(-koff * dt)
            if np.random.random() < p_off:
                log_xl_event("unbind", current_step, sim_time, xl.fi+1, xl.ni, xl.fj+1, xl.nj, L, xl.L0, F_sim, koff)
                dead.append(key)
        for key in dead:
            self.links[key].remove_from_view(self.view)
            del self.links[key]

    def redraw(self, fibers):
        if not DRAW_CROSSLINKS:
            return
        for xl in self.links.values():
            xl.redraw(fibers)

    def fiber_involved(self, fi):
        for xl in self.links.values():
            if xl.fi == fi or xl.fj == fi:
                return True
        return False

# ==================== Voronoi loaders ====================

def _is_int_token(tok):
    try:
        int(tok); return True
    except Exception:
        return False

def _to_float_list(tokens):
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except Exception:
            return None
    return out

def load_voronoi_vertices(path):
    verts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            s = s.replace(",", " ")
            toks = [t for t in s.split() if t]
            if not toks: continue
            floats = _to_float_list(toks)
            if floats is None and _is_int_token(toks[0]):
                floats = _to_float_list(toks[1:])
            if floats is None: continue
            if len(floats) == 2:
                x, y = floats; z = 0.0
            elif len(floats) >= 3:
                x, y, z = floats[:3]
            else:
                continue
            verts.append([x, y, z])
    if not verts:
        raise ValueError(f"No vertices parsed from {path}")
    return np.array(verts, dtype=np.float64)

def load_voronoi_edges(path, n_vertices):
    edges = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            s = s.replace(",", " ")
            toks = [t for t in s.split() if t]
            if not toks: continue
            if len(toks) >= 2 and _is_int_token(toks[-1]) and _is_int_token(toks[-2]):
                i = int(toks[-2]); j = int(toks[-1])
            else:
                continue
            edges.append((i, j))
    if not edges:
        raise ValueError(f"No edges parsed from {path}")

    mins = min(min(i, j) for i, j in edges)
    maxs = max(max(i, j) for i, j in edges)
    one_based = (mins >= 1) and (maxs <= n_vertices)
    if one_based:
        edges = [(i - 1, j - 1) for i, j in edges]

    seen = set(); uniq = []
    for i, j in edges:
        if not (0 <= i < n_vertices and 0 <= j < n_vertices):
            continue
        key = (i, j) if i <= j else (j, i)
        if key in seen: continue
        seen.add(key); uniq.append(key)
    return uniq

def build_fibers_from_voronoi(vertices_path, edges_path, scale=1.0, center=True, limit=None):
    verts = load_voronoi_vertices(vertices_path)
    if center:
        centroid = np.mean(verts, axis=0); verts = verts - centroid
    if scale != 1.0:
        verts = verts * float(scale)
    edges = load_voronoi_edges(edges_path, verts.shape[0])
    if limit is not None:
        edges = edges[:int(limit)]
    print(f"[Voronoi] Loaded {verts.shape[0]} vertices, {len(edges)} edges (centered={center}, scale={scale}).")

    # Precompute collagen cross-section + second moment
    r = 0.5 * DF_COLLAGEN
    A = np.pi * r**2                 # area
    I = (np.pi * r**4) / 4.0         # second moment of area

    fibs = []
    for (i, j) in edges:
        p1 = verts[i]; p2 = verts[j]
        f = ElasticFiberFEM(start_point=p1, end_point=p2)

        # Use the fiber’s mean segment length for stiffness mapping
        mean_L = float(np.mean(np.maximum(1e-12, f.rest_lengths)))

        # Discrete equivalents from continuum beam/rod theory
        f.axial_k   = EF_COLLAGEN * A / mean_L
        f.bending_k = EF_COLLAGEN * I / (mean_L**3)

        fibs.append(f)

    return fibs

# ==================== PyQt + Simulation ====================

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.setWindowTitle('Integrin Springs (Compression + Preload at Bind) + Catch–Slip + Fiber–Fiber Links (Voronoi, free-end fibers, no L0 shrink)')
view.setGeometry(0, 110, 1280, 800)
view.setCameraPosition(distance=2.0, elevation=20, azimuth=45)
view.show()

# overlay labeler
overlay = OverlayLabeler(view)

# Cells (purely visual now)
cell_counter = 0
def add_cell_mesh(center, radius=CELL_RADIUS, color=(0.3, 0.6, 1.0, 0.25)):
    global cell_counter
    center = np.array(center, dtype=np.float64)
    mesh = gl.GLMeshItem(
        meshdata=gl.MeshData.sphere(rows=24, cols=24, radius=float(radius)),
        smooth=True, color=color, shader='shaded'
    )
    mesh.translate(*center)
    view.addItem(mesh)
    cell_counter += 1
    label_pos = center + np.array([0.0, 0.0, radius * 1.2])
    overlay.set_label(f"cell:{cell_counter}", str(cell_counter), label_pos)

# initial cell
add_cell_mesh(CELL_CENTER)

manager = IntegrinManager(view)
manager.populate_integrins_on_sphere(N=NUM_INTEGRINS, center=CELL_CENTER, radius=CELL_RADIUS)

# Voronoi fibers
if not os.path.exists(VOR_VERTICES_PATH) or not os.path.exists(VOR_EDGES_PATH):
    raise FileNotFoundError(
        f"Voronoi files not found. Expected:\n  {os.path.abspath(VOR_VERTICES_PATH)}\n  {os.path.abspath(VOR_EDGES_PATH)}"
    )

fibers = build_fibers_from_voronoi(
    VOR_VERTICES_PATH,
    VOR_EDGES_PATH,
    scale=VORONOI_SCALE,
    center=CENTER_VORONOI,
    limit=FIBER_LIMIT
)

# Number and label all fibers (1-indexed labels)
for idx, f in enumerate(fibers, start=1):
    f.index = idx
    overlay.set_label(f"fiber:{idx}", str(idx), f._midpoint())

crosslinks = CrosslinkManager(view)

TOTAL_STEPS = 15
STEP_INTERVAL_MS = 100
REACTIVATE = True

current_step = 0
step_timer = QtCore.QTimer()

def fibers_bbox():
    all_xyz = np.vstack([f.x for f in fibers])
    return np.min(all_xyz, axis=0), np.max(all_xyz, axis=0)

CELL_CENTERS = [CELL_CENTER.copy()]

def add_random_cell_with_integrins(max_attempts=50, min_gap_factor=1.05):
    """Add a new cell near a random fiber node without overlapping existing cells."""
    global CELL_CENTERS

    for attempt in range(max_attempts):
        # pick random fiber and node
        f = random.choice(fibers)
        node_idx = random.randint(0, f.N - 1)
        node_pos = f.x[node_idx]

        # random outward direction
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)

        # proposed center
        center = node_pos + direction * (CELL_RADIUS * 0.7)

        # check overlap
        ok = True
        for c_prev in CELL_CENTERS:
            dist = np.linalg.norm(center - c_prev)
            if dist < (2 * CELL_RADIUS * min_gap_factor):
                ok = False
                break

        if ok:
            # add cell and record center
            add_cell_mesh(center)
            manager.populate_integrins_on_sphere(N=NUM_INTEGRINS,
                                                 center=center,
                                                 radius=CELL_RADIUS)
            CELL_CENTERS.append(center)
            return  # success → exit function

    print(f"[WARN] Could not place non-overlapping cell after {max_attempts} attempts.")

def frame_1():
    print("Frame 1: Waiting...")
    capture_frame()

def frame_2():
    print("Frame 2: Set up logs, draw fibers, activate integrins...")
    with open(BOND_EVENTS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event","step","time_s","integrin","fiber_idx","node_idx",
                    "L","L0","extension_e","F_sim","F_phys","koff","lifetime_s"])
    with open(BOND_SERIES_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step","time_s","integrin","fiber_idx","node_idx",
                    "L","L0","extension_e","F_sim","F_phys","koff"])
    with open(XL_EVENTS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event","step","time_s","fiber_i","node_i","fiber_j","node_j","L","L0","F","koff"])
    with open(XL_SERIES_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step","time_s","fiber_i","node_i","fiber_j","node_j","L","L0","F","koff"])
    with open(INTEGRIN_FORCE_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step","time_s","integrin","fiber_idx","node_idx",
                    "extension_e","F_sim","F_phys"])

    for f in fibers:
        f.line.setData(pos=f.x)

    open(INTEGRIN_DIST_CSV, "w").close()
    manager.activate_near_fibers(fibers, threshold=None, save_path=INTEGRIN_DIST_CSV)
    capture_frame()

def frame_3():
    with open("fiber_deformation3.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["Step", "Fiber", "Node", "X", "Y", "Z"])
    print("Starting multi-step simulation...")
    step_timer.setInterval(STEP_INTERVAL_MS)
    step_timer.timeout.connect(simulation_step)
    step_timer.start()

def fiber_has_other_forces(fib_obj):
    fi = fibers.index(fib_obj)
    for integ in manager.integrins.values():
        if integ.state == 'active' and integ.attachment is not None:
            f_attached, _ = integ.attachment
            if f_attached is fib_obj:
                return True
    if crosslinks.fiber_involved(fi):
        return True
    return False

def simulation_step():
    global current_step, sim_time
    try:
        dt = STEP_INTERVAL_MS / 1000.0

        if ADD_NEW_CELL_EACH_STEP:
            add_random_cell_with_integrins()

        crosslinks.formation_step(fibers, dt)

        if REACTIVATE:
            manager.activate_near_fibers(fibers, threshold=None, save_path=INTEGRIN_DIST_CSV)

        active = [i for i in manager.integrins.values() if i.state == 'active']
        print(f"Step {current_step}: active integrins = {len(active)}, fiber-fiber links = {len(crosslinks.links)}")

        # log BEFORE (Fiber 1-indexed)
        with open("fiber_deformation3.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            for j, f in enumerate(fibers):
                jf = j + 1
                for i, node in enumerate(f.x):
                    w.writerow([f"STEP_{current_step}_BEFORE", jf, i, *node])

        still_bound = []
        for integ in active:
            if integ.attachment is None:
                continue
            fib, idx = integ.attachment
            attach_pt = fib.x[idx]
            pivot = integ._pivot_point()

            d_vec_top = attach_pt - pivot
            L_top = float(np.linalg.norm(d_vec_top))
            if L_top < 1e-9:
                integ._draw_active_hinged_to(attach_pt)
                continue

            # Overstretch guard relative to geometric max
            if L_top > integ.top_max:
                fib_idx = fibers.index(fib) + 1
                e_tmp = abs(L_top - integ.L0)
                F_sim_tmp = SPRING_K_SIM * e_tmp
                e_nm_tmp = e_tmp * 1e9
                F_phys_tmp = (K_PN_PER_NM * e_nm_tmp) * 1e-12
                koff_tmp = Integrin.catch_slip_off_rate(F_phys_tmp)
                integ.unbind()
                fib.reset_to_rest(alpha=1.0)
                log_bond_event("unbind", current_step, sim_time, integ.id, fib_idx, idx,
                               L_top, integ.L0, e_tmp, F_sim_tmp, F_phys_tmp, koff_tmp,
                               lifetime=integ.bound_time)
                print(f"Step {current_step}: Integrin {integ.id} unbound due to overstretch (L_top={L_top:.3e} > top_max={integ.top_max:.3e})")
                continue  # don't process further

            # Unit vector from pivot to attachment
            u_top = d_vec_top / L_top

            # Hinge geometry for drawing
            integ.length = (integ.inactive_length * 0.5) + min(L_top, integ.top_max)
            integ.bound_time += dt

            # --- Compression-style extension: |L_top - L0| ---
            e = abs(L_top - integ.L0)
            F_sim  = SPRING_K_SIM * e
            e_nm   = e * 1e9
            F_phys = (K_PN_PER_NM * e_nm) * 1e-12
            koff   = Integrin.catch_slip_off_rate(F_phys)

            # --- force-based rupture cap (based on magnitude) ---
            F_CAP = 260e-12  # 350 pN rupture threshold

            if F_phys > F_CAP:
                fib_idx = fibers.index(fib) + 1

                # Unbind and reset
                integ.unbind()
                fib.reset_to_rest(alpha=1.0)

                # Log the unbinding as a 'force_unbind' event
                log_bond_event("force_unbind", current_step, sim_time, integ.id, fib_idx, idx,
                               L_top, integ.L0, e, F_sim, F_phys, koff, lifetime=integ.bound_time)

                print(f"Step {current_step}: Integrin {integ.id} unbound due to exceeding F_cap ({F_phys*1e12:.2f} pN > {F_CAP*1e12:.2f} pN)")
                continue  # skip further updates for this integrin

            fib_idx = fibers.index(fib) + 1
            log_bond_state(current_step, sim_time, integ.id, fib_idx, idx,
                           L_top, integ.L0, e, F_sim, F_phys, koff)
            log_integrin_force(current_step, sim_time, integ.id, fib_idx, idx,
                               e, F_sim, F_phys)

            # Probabilistic unbinding BEFORE applying compression force
            p_unbind = 1.0 - np.exp(-koff * dt)
            if UNBIND_BEFORE_FORCE and random.random() < p_unbind:
                integ.unbind()
                fib.reset_to_rest(alpha=1.0)
                log_bond_event("unbind", current_step, sim_time, integ.id, fib_idx, idx,
                               L_top, integ.L0, e, F_sim, F_phys, koff, lifetime=integ.bound_time)
                print(f"Step {current_step}: Integrin {integ.id} unbound probabilistically (catch-slip, koff={koff:.3e}, F_phys={F_phys:.3e} N, p_unbind={p_unbind:.3f})")
                continue

            # Redraw integrin to current attachment point
            integ.remove_integrin()
            integ._draw_active_hinged_to(attach_pt)

            # --- Apply COMPRESSION force to fiber ---
            # Direction from attachment toward pivot is -u_top (push fiber closer to pivot)
            if F_sim > 0:
                fib.apply_external_force(idx, -F_sim * u_top)

            still_bound.append((integ, fib, idx))

        crosslinks.force_and_break_step(fibers, dt)

        for f in fibers:
            f.update()  # updates fiber label positions, too

        crosslinks.redraw(fibers)

        for integ, fib, idx in still_bound:
            attach_post = fib.x[idx]
            integ.remove_integrin()
            integ._draw_active_hinged_to(attach_post)
            
        fibers_with_active_bonds = set()
        for integ in manager.integrins.values():
            if integ.state == "active" and integ.attachment is not None:
                fibers_with_active_bonds.add(integ.attachment[0])

        # Reset only the free ones
        for f in fibers:
            if hasattr(f, "x_rest") and f not in fibers_with_active_bonds:
                f.reset_to_rest(alpha=1.0)

        # log AFTER (Fiber 1-indexed)
        with open("fiber_deformation3.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            for j, f in enumerate(fibers):
                jf = j + 1
                for i, node in enumerate(f.x):
                    w.writerow([f"STEP_{current_step}_AFTER", jf, i, *node])

        capture_frame()

        sim_time += dt
        current_step += 1
        if current_step >= TOTAL_STEPS:
            step_timer.stop()
            print("Simulation loop complete.")
            save_video()
            # Analysis / plotting
            plot_integrin_forces()
            plot_fiber_displacement_over_time()
            plot_force_vs_displacement()

    except Exception as e:
        print("ERROR in simulation_step:", repr(e))
        step_timer.stop()

# ==================== PLOTTING / ANALYSIS ====================

def plot_integrin_forces(csv_path="integrin_forces.csv", out_path="integrin_force_over_time.png"):
    # --- Load ---
    df = pd.read_csv(csv_path)
    df["step"] = df["step"].astype(int)
    df["time_s"] = df["time_s"].astype(float)
    df["F_sim"] = df["F_sim"].astype(float)
    df["F_phys"] = df["F_phys"].astype(float)

    # --- Sort chronologically ---
    df = df.sort_values(["integrin", "step"])

    # --- Group by integrin ---
    plt.figure(figsize=(12, 8))
    for integ_id, g in df.groupby("integrin"):
        plt.plot(g["time_s"], g["F_phys"] * 1e12, label=str(integ_id))  # convert N→pN

    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Force per Integrin (pN)")
    plt.title("Integrin Force vs Time (active integrins)")
    plt.legend(title="Integrin ID", bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[✓] Saved integrin force plot to {out_path}")

def plot_fiber_displacement_over_time(
    deform_csv="fiber_deformation3.csv",
    out_path="fiber_displacement_over_time_top10.png",
    top_n=10
):
    """
    Plots displacement over time for the top-N most deformed fibers.
    'Most deformed' is defined as having the largest cumulative mean
    node displacement over all steps.
    """

    # --- Load deformation data ---
    df = pd.read_csv(deform_csv)

    # Parse step index (e.g. "STEP_3_BEFORE" or "STEP_3_AFTER")
    def step_to_int(s):
        m = re.search(r"STEP_(\d+)_", str(s))
        return int(m.group(1)) if m else np.nan

    df["step"] = df["Step"].map(step_to_int).astype("Int64")
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)

    # Separate BEFORE and AFTER positions
    before = df[df["Step"].str.contains("_BEFORE")].copy()
    after  = df[df["Step"].str.contains("_AFTER")].copy()

    # Common keys
    key = ["step", "Fiber", "Node"]
    b = before.set_index(key)[["X", "Y", "Z"]]
    a = after.set_index(key)[["X", "Y", "Z"]]
    common = b.index.intersection(a.index)
    if len(common) == 0:
        print("[WARN] No matching BEFORE/AFTER entries found for displacement.")
        return

    # --- Compute displacement magnitude per node ---
    disp = np.linalg.norm(a.loc[common].values - b.loc[common].values, axis=1)
    disp_df = pd.DataFrame({
        "step": [i[0] for i in common],
        "Fiber": [i[1] for i in common],
        "Node": [i[2] for i in common],
        "displacement": disp
    })

    # --- Average displacement per fiber per step ---
    fiber_disp = (
        disp_df.groupby(["Fiber", "step"])["displacement"]
        .mean()
        .reset_index()
    )

    if fiber_disp.empty:
        print("[WARN] No displacement data available for fibers.")
        return

    # --- Compute cumulative displacement per fiber (to rank top-N) ---
    fiber_total = (
        fiber_disp.groupby("Fiber")["displacement"]
        .sum()
        .sort_values(ascending=False)
    )

    top_fibers = list(fiber_total.head(top_n).index)
    print(f"[INFO] Top {len(top_fibers)} fibers by cumulative displacement:", top_fibers)

    fiber_disp_top = (
        fiber_disp[fiber_disp["Fiber"].isin(top_fibers)]
        .sort_values(["Fiber", "step"])
    )

    # --- Plot ---
    plt.figure(figsize=(12, 8))
    for fiber_id in sorted(top_fibers):
        g = fiber_disp_top[fiber_disp_top["Fiber"] == fiber_id]
        if g.empty:
            continue
        plt.plot(g["step"], g["displacement"] * 1e6, label=f"Fiber {fiber_id}")  # m → µm

    plt.xlabel("Simulation Step")
    plt.ylabel("Mean Fiber Displacement (µm)")
    plt.title(f"Fiber Displacement vs Time (Top {len(top_fibers)} Most Deformed Fibers)")
    plt.legend(title="Fiber ID", bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[✓] Saved top-{len(top_fibers)} fiber displacement plot to {out_path}")

def plot_force_vs_displacement(
    deform_csv="fiber_deformation3.csv",
    bond_csv="bond_timeseries2.csv",
    out_path="force_vs_displacement_per_fiber_step.png"
):
    """
    Plots the relationship between mean integrin force and mean fiber displacement
    per fiber-step. Each point represents one (Fiber, step) pair.
    """

    # --- Load deformation data and compute mean displacement per fiber-step ---
    df = pd.read_csv(deform_csv)

    def step_to_int(s):
        m = re.search(r"STEP_(\d+)_", str(s))
        return int(m.group(1)) if m else np.nan

    df["step"] = df["Step"].map(step_to_int).astype("Int64")
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)

    before = df[df["Step"].str.contains("_BEFORE")].copy()
    after  = df[df["Step"].str.contains("_AFTER")].copy()

    key = ["step", "Fiber", "Node"]
    b = before.set_index(key)[["X", "Y", "Z"]]
    a = after.set_index(key)[["X", "Y", "Z"]]
    common = b.index.intersection(a.index)
    if len(common) == 0:
        print("[WARN] No matching BEFORE/AFTER entries found for force–displacement analysis.")
        return

    disp = np.linalg.norm(a.loc[common].values - b.loc[common].values, axis=1)
    disp_df = pd.DataFrame({
        "step": [i[0] for i in common],
        "Fiber": [i[1] for i in common],
        "Node": [i[2] for i in common],
        "displacement": disp
    })

    disp_mean = (
        disp_df.groupby(["Fiber", "step"])["displacement"]
        .mean()
        .reset_index()
    )

    # --- Load bond timeseries and compute mean force per fiber-step ---
    bond = pd.read_csv(bond_csv)
    if bond.empty:
        print("[WARN] bond_timeseries2.csv is empty; cannot compute force–displacement relationship.")
        return

    bond["step"] = bond["step"].astype(int)
    bond["Fiber"] = bond["fiber_idx"].astype(int)
    # Mean physical force (N) per fiber-step
    force_mean = (
        bond.groupby(["Fiber", "step"])["F_phys"]
        .mean()
        .reset_index()
    )

    # --- Merge on (Fiber, step) ---
    merged = pd.merge(force_mean, disp_mean, on=["Fiber", "step"], how="inner")
    if merged.empty:
        print("[WARN] No overlapping (Fiber, step) entries between force and displacement.")
        return

    # Convert to nice units: displacement → µm, force → pN
    merged["displacement_um"] = merged["displacement"] * 1e6
    merged["force_pN"] = merged["F_phys"] * 1e12

    # --- Scatter plot ---
    plt.figure(figsize=(8, 6))
    plt.scatter(merged["displacement_um"], merged["force_pN"], alpha=0.6)
    plt.xlabel("Mean Fiber Displacement per Step (µm)")
    plt.ylabel("Mean Integrin Force per Step (pN)")
    plt.title("Force–Displacement Relationship (per Fiber-Step)")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[✓] Saved force–displacement scatter to {out_path}")

def save_video():
    print("Saving video to integrin_simulation3.mp4...")
    imageio.mimsave("integrin_simulation3.mp4", frame_images, fps=2)

QtCore.QTimer.singleShot(1000, frame_1)
QtCore.QTimer.singleShot(2500, frame_2)
QtCore.QTimer.singleShot(4000, frame_3)

app.exec_()
