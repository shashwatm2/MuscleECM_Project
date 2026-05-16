import numpy as np
import csv
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import imageio
import random

frame_images = []
frame_count = 0

CELL_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
CELL_RADIUS = 0.4

NUM_INTEGRINS = 30
NUM_FIBERS = 25
ADD_NEW_CELL_EACH_STEP = True

SPRING_K_SIM   = 5.5e6      # N/m (top segment only)
K_PN_PER_NM    = 2e-5       # pN per nm for off-rate model

# preload: constant force at first attachment
BASE_PRELOAD_PN = 200.0     # pN, you can change this

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

def angle_between(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class IntegrinLineManager:
    def __init__(self, view):
        self.view = view
        self.lines = []

    def clear(self):
        for l in self.lines:
            try:
                self.view.removeItem(l)
            except:
                pass
        self.lines = []

    def add_line(self, p1, p2):
        pts = np.array([p1, p2])
        line = gl.GLLinePlotItem(pos=pts, color=(0,1,0,1), width=2, antialias=True)
        self.view.addItem(line)
        self.lines.append(line)

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

        # For a flat substrate at z=0, make integrins point straight up.
        self.n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Build two in-plane tangent directions (for dimer offset and head orientation)
        ref = np.array([1.0, 0.0, 0.0])  # any non-colinear vector works here
        self.t1 = _unit(np.cross(ref, self.n))   # will be ~[0, -1, 0]
        self.t2 = _unit(np.cross(self.n, self.t1))  # will be ~[1, 0, 0]

        self.label_num = label_num  # integer
        self.add_inactive_integrin()

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

    def _pivot_point(self):
        return self.base_position + self.n * self.inactive_length

    def add_inactive_integrin(self):

        # self.items.clear()
        # x, y, z = self.base_position
        # for i, color in enumerate(self.colors):
        #     cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20), smooth=True, color=color, shader='shaded')
        #     cyl.scale(self.radius, self.radius, self.length / 2)
        #     cyl.translate(x + (i - 0.5) * self.radius * 3, y, z)
        #     self.items.append(cyl)
        # for i, color in enumerate(self.colors):
        #     top = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20), smooth=True, color=color, shader='shaded')
        #     top.rotate(90, 1, 0, 0)
        #     top.scale(self.radius, self.radius, self.length / 2)
        #     top.translate(x + (i - 0.5) * self.radius * 3, y, z + self.length / 2)
        #     self.items.append(top)
        # self.view_integrin()

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
    
    def update_active_geometry(self):
        if self.state != 'active' or self.attachment is None:
            return

        fiber, idx = self.attachment
        attach_pt = fiber.x[idx].copy()
        self._draw_active_hinged_to(attach_pt)

class IntegrinManager:
    def __init__(self, view):
        self.view = view
        self.integrins = {}
        self.cell_radius = None
        self.perimeter_item = None

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

        self.cell_radius = radius
        self.draw_perimeter(circle_z = z + 0.001, segments = 180)

    def draw_perimeter(self, circle_z=0.001, segments=180):
        theta = np.linspace(0, 2*np.pi, segments, endpoint=True)
        xs = self.cell_radius * np.cos(theta)
        ys = self.cell_radius * np.sin(theta)
        zs = np.full_like(xs, circle_z)
        pts = np.vstack([xs, ys, zs]).T.astype(np.float32)

        if self.perimeter_item is not None:
            try:
                self.view.removeItem(self.perimeter_item)
            except:
                pass

        self.perimeter_item = gl.GLLinePlotItem(pos=pts, color=(1,1,1,1), width=2, antialias=True, mode='line_strip')
        pts_closed = np.vstack([pts, pts[0:1]])
        self.perimeter_item.setData(pos=pts_closed)
        self.view.addItem(self.perimeter_item)

    def activate_near_fibers(self, fibers, threshold=None, save_path=None):
        global current_step, sim_time
        any_activated = False
        rows = []

        for integ in self.integrins.values():

            # Already active → skip activation attempt
            if integ.state == 'active':
                tip_now = integ.get_tip_position()
                rows.append([integ.id, *tip_now, 0.0, False])
                continue

            # Find nearest fiber point
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

            if best is None:
                rows.append([integ.id, *tip_pre, 0.0, False])
                continue

            # Threshold / reach distance
            reach = 2 * integ.top_max if threshold is None else float(threshold)

            if min_dist >= reach:
                rows.append([integ.id, *tip_pre, min_dist, False])
                continue

            f, idx, pt = best

            # Z-gate (fiber must be above the pivot)
            pivot = integ._pivot_point()
            if pt[2] < pivot[2]:
                rows.append([integ.id, *tip_pre, min_dist, False])
                continue

            # Binding length constraint
            L_bind_top = float(np.linalg.norm(pt - pivot))
            if L_bind_top > integ.top_max:
                rows.append([integ.id, *tip_pre, min_dist, False])
                continue

            # Ensure attachment node exists on the fiber
            attach_idx, attach_pt = f.ensure_attachment_node(pt)
            if attach_idx is None:
                rows.append([integ.id, *tip_pre, min_dist, False])
                continue

            # Activation + preload extension
            e0_nm = BASE_PRELOAD_PN / K_PN_PER_NM   # preload nm
            e0 = e0_nm * 1e-9                       # convert to meters
            integ.L0 = max(L_bind_top - e0, 1e-9)

            integ.switch_to_active(attach_pt)
            integ.attachment = (f, attach_idx)

            any_activated = True
            rows.append([integ.id, *attach_pt, min_dist, True])

        return any_activated

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

    def ensure_attachment_node(self, pos, min_sep=1e-8):
        idx, pt = self.get_closest_point(pos)
        if idx is None or pt is None:
            return None, None
        if np.linalg.norm(pt - self.x[idx])   <= min_sep: return idx,     self.x[idx]
        if np.linalg.norm(pt - self.x[idx+1]) <= min_sep: return idx + 1, self.x[idx+1]
        new_idx = self.insert_node(idx, pt)
        return new_idx, self.x[new_idx]

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
    
    def insert_node(self, segment_idx, position):
        self.x = np.insert(self.x, segment_idx + 1, position, axis=0)
        self.v = np.insert(self.v, segment_idx + 1, 0.0, axis=0)
        self.f_ext = np.insert(self.f_ext, segment_idx + 1, 0.0, axis=0)

        self.N = len(self.x)

        total_length = np.sum(np.linalg.norm(np.diff(self.x, axis=0), axis=1))
        self.rest_length = total_length / (self.N - 1)

        self.line.setData(pos=self.x)

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
                # log_xl_event("bind", current_step, sim_time, fi+1, ni, fj+1, nj, d, d, 0.0, KFF_OFF0)

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
                # log_xl_event("unbind", current_step, sim_time, xl.fi+1, xl.ni, xl.fj+1, xl.nj, L, xl.L0, F_sim, koff)
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

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.setWindowTitle('Integrin Activation and Multiple Elastic Fibers')
view.setGeometry(0, 110, 1280, 800)
view.show()
view.setCameraPosition(distance=2.0, elevation=20, azimuth=45)

manager = IntegrinManager(view)
manager.populate_integrins(N=NUM_INTEGRINS, radius=CELL_RADIUS, z=0)

line_manager = IntegrinLineManager(view)

fibers = []
for _ in range(NUM_FIBERS):
    p1 = np.random.uniform(low=[-1*CELL_RADIUS, -1*CELL_RADIUS, 0.045], high=[CELL_RADIUS, CELL_RADIUS, 0.065])
    p2 = np.random.uniform(low=[-1*CELL_RADIUS, -1*CELL_RADIUS, 0.045], high=[CELL_RADIUS, CELL_RADIUS, 0.065])
    fibers.append(ElasticFiberFEM(start_point=p1, end_point=p2))

crosslinks = CrosslinkManager(view)

TOTAL_STEPS = 5
STEP_INTERVAL_MS = 500
FORCE_GAIN = 1e7
REACTIVATE = True

current_step = 0
step_timer = QtCore.QTimer()

def frame_1():
    print("Frame 1: Waiting...")
    capture_frame()

def frame_2():
    print("Frame 2: Drawing fibers and activating integrins...")
    for f in fibers:
        f.line.setData(pos=f.x)
    manager.activate_near_fibers(fibers, threshold=None)
    capture_frame()

def frame_3():
    with open("fiber_deformation2.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Fiber", "Node", "X", "Y", "Z"])
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
    global current_step
    try:
        dt = STEP_INTERVAL_MS / 1000.0
        
        if REACTIVATE:
            manager.activate_near_fibers(fibers, threshold=None, save_path=None)

        crosslinks.formation_step(fibers, dt)

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
                    best_fiber.insert_node(best_idx, tip)
                    best_fiber.apply_external_force(best_idx + 1, vec * FORCE_GAIN)

        # -------------------------------
        # CONTINUOUS SPRING PULLING
        # -------------------------------
        for integ in manager.integrins.values():
            if integ.state == 'active' and integ.attachment is not None:
                fiber, idx = integ.attachment

                # Safety check: node index must still exist
                if idx < 0 or idx >= fiber.N:
                    continue

                attach_pt = fiber.x[idx]
                pivot = integ._pivot_point()

                vec = attach_pt - pivot
                dist = np.linalg.norm(vec)
                if dist < 1e-12:
                    continue

                # Direction of pulling: toward pivot
                direction = -vec / dist

                # Hookean extension
                extension = dist - integ.L0
                F = integ.k_spring * extension

                # Apply to fiber
                fiber.apply_external_force(idx, F * direction)

        crosslinks.force_and_break_step(fibers, dt)
        
        for f in fibers:
            f.update()

        crosslinks.redraw(fibers)

        # Update all active integrins so they rotate toward their bound fiber point
        for integ in manager.integrins.values():
            if integ.state == 'active':
                integ.update_active_geometry()

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

QtCore.QTimer.singleShot(5000, frame_1)
QtCore.QTimer.singleShot(5000, frame_2)
QtCore.QTimer.singleShot(5000, frame_3)

app.exec_()
