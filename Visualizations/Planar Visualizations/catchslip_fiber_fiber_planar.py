import numpy as np
import csv
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import imageio
import random

# ==================== Config: Spring + Catch–Slip (integrins) ====================

# Mechanics: spring force actually applied to fibers (top-segment only)
SPRING_K_SIM = 1e8      # N/m
L0_SHRINK_RATE = 1e-5     # m/s; >0 builds tension over time

# Biophysics for off-rate ONLY: use pN/nm-scale "physical" spring
K_PN_PER_NM = 0.002      # pN per nm effective stiffness for k_off(F)

# Catch–slip (Kong et al. 2009) — parameters in pN converted to N here
# k_off(F) = a1*exp(-((F-b1)/c1)**2) + a2*exp(-((F-b2)/c2)**2)
CATCHSLIP_A1 = 2.2          # 1/s
CATCHSLIP_A2 = 1.2          # 1/s
CATCHSLIP_B1 = 29.9e-12     # N  (29.9 pN)
CATCHSLIP_B2 = 16.2e-12     # N  (16.2 pN)
CATCHSLIP_C1 = 8.4e-12      # N  (8.4 pN)
CATCHSLIP_C2 = 37.8e-12     # N  (37.8 pN)

UNBIND_BEFORE_FORCE = True  # sample unbinding before pushing the fiber this step

# ==================== Config: Fiber↔Fiber crosslinks (paper §2.3) ====================

# On/off parameters (paper defaults)
KFF_ON   = 0.01         # s^-1  (chemical association rate)
KFF_OFF0 = 1.0e-4          # s^-1  (chemical dissociation rate at zero force)
BELL_DX  = 0.5e-9          # m     (Bell parameter Δx)
TEMP_K   = 300.0           # K     (absolute temperature)
KB       = 1.380649e-23    # J/K   (Boltzmann)
KB_T     = KB * TEMP_K     # J

# Mechanics of the crosslink element (spring surrogate for a short beam)
CROSSLINK_K_SIM = 1e6      # N/m   (stiff, but below integrin spring to avoid explosions)
CROSSLINK_RANGE = 0.01    # m     (critical capture distance for on-rate check)
CROSSLINK_STRIDE = 1       # only consider every Nth node to keep pair checks cheap
DRAW_CROSSLINKS = True

# ----- Crosslink types (ionic vs covalent) -----
IONIC_FRACTION = 0.5  # probability of ionic link at formation

# Colors
IONIC_COLOR   = (1, 0, 0, 1)   # red
COVALENT_COLOR= (0, 1, 0, 1)   # green

# Type multipliers (relative to base constants KFF_OFF0, BELL_DX, CROSSLINK_K_SIM)
IONIC_K_MULT        = 0.6
IONIC_KOFF0_MULT    = 5.0
IONIC_DX_MULT       = 1.5

COVALENT_K_MULT     = 1.4
COVALENT_KOFF0_MULT = 0.0
COVALENT_DX_MULT    = 0.7

# --- Numerical safety for Bell off-rate ---
KFF_FORCE_CLAMP_PN = 500.0          # pN cap used inside exp()
KFF_FORCE_CLAMP_N  = KFF_FORCE_CLAMP_PN * 1e-12  # N
KFF_EXP_CLAMP      = 60.0            # cap exponent to avoid overflow

# --- Effective stiffness for off-rate only (pN/nm), decoupled from sim force ---
KFF_PN_PER_NM      = 1e-3            # smaller -> more stable unbinding rates


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

# ==================== Bond logging (integrins) ====================

BOND_EVENTS_CSV = "bond_events2.csv"
BOND_SERIES_CSV = "bond_timeseries2.csv"
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

# ==================== Crosslink logging (fiber↔fiber) ====================

XL_EVENTS_CSV = "crosslink_events2.csv"
XL_SERIES_CSV = "crosslink_timeseries2.csv"

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

# ==================== Integrin + Manager ====================

class Integrin:
    def __init__(self, view, id, base_position, length=0.004, radius=0.0003,
                 colors=[(1, 0, 0, 1), (0, 0, 1, 1)]):
        self.view = view
        self.id = id
        self.base_position = np.array(base_position, dtype=np.float64)  # fixed base (z=0)
        self.length = float(length)      # current total length proxy (used for tip calc when inactive)
        self.radius = float(radius)
        self.colors = colors

        self.state = 'inactive'
        self.items = []

        self.inactive_length = float(length)       # full integrin nominal length (0.004)
        self.inactive_top_length = self.inactive_length * 0.5  # top half nominal (0.002)
        self.active_length   = 2.0 * float(length)  # total when active (0.008)

        # --- length limits ---
        # total length must never exceed active_length (0.008 by default)
        # bottom (fixed) = inactive_length/2; so the max top length is:
        self.top_max = self.inactive_length

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
                           scale=(self.radius, self.radius, self.inactive_length))
        # top horizontal heads at pivot height
        for i, color in enumerate(self.colors):
            top = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(top,
                           translate=(x + (i - 0.5) * self.radius * 3.0, y, z + self.inactive_length),
                           rotate=(90, 1, 0, 0),
                           scale=(self.radius, self.radius, self.inactive_length))

    def _pivot_point(self):
        # Hinge between bottom and top pairs
        return self.base_position + np.array([0.0, 0.0, self.inactive_length], dtype=np.float64)

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

        # top segment geometry (CLAMPED)
        to_target = np.array(target_pt, dtype=np.float64) - pivot
        L_top_raw = float(np.linalg.norm(to_target))
        if L_top_raw < 1e-9:
            angle_deg = 0.0
            axis = np.array([1.0, 0.0, 0.0])
            L_top = 0.0
            u = np.array([0.0, 0.0, 1.0])
        else:
            u = to_target / L_top_raw
            # ---- NEVER draw beyond max top reach ----
            L_top = min(L_top_raw, self.top_max)  # <<< NEW
            # orientation from +Z to u (same as before)
            z_axis = np.array([0.0, 0.0, 1.0])
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
        L_bottom = self.inactive_length
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                cyl,
                translate=(float(x + offsets[i][0]), float(y + offsets[i][1]), float(z)),
                scale=(self.radius, self.radius, L_bottom)
            )

        # ---- Top pair: rotate from the pivot toward the target (clamped length) ----
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
        self.length = L_bottom + L_top  # <= will never exceed 0.008 now  # <<< CHANGED

    # ---- state changes ----
    def switch_to_active(self, attach_pt=None):
        if self.state != 'inactive':
            return
        if attach_pt is not None:
            pivot = self._pivot_point()
            L_bind_top = float(np.linalg.norm(np.array(attach_pt, dtype=np.float64) - pivot))
            # ---- refuse binding if top needed length exceeds max ----
            if L_bind_top > self.top_max:  # <<< NEW
                return  # stay inactive; out of reach
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
        OPTION B:
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

    def activate_near_fibers(self, fibers, threshold=None, save_path="integrin_distances3.csv"):
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
                    # Compute pivot→attach top length and enforce hard cap  # <<< NEW
                    pivot = integ._pivot_point()
                    L_bind_top = float(np.linalg.norm(attach_pt - pivot))
                    if L_bind_top > integ.top_max:
                        # too far to reach even when fully active; skip binding
                        rows.append([integ.id, *integ.get_tip_position(), min_dist, False])
                        continue

                    integ.switch_to_active(attach_pt)
                    integ.attachment = (f, attach_idx)
                    activated = True
                    any_activated = True

                    # ---- logging: bind event (TOP SPRING) ----
                    fib_idx = fibers.index(f)
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

# ==================== Fiber↔Fiber Crosslinks ====================

class Crosslink:
    """Transient link between fiber i:ni and fiber j:nj; initially stress-free (L0 at formation)."""
    def __init__(self, view, fi, ni, fj, nj, L0, k=CROSSLINK_K_SIM, xtype='ionic'):
        self.fi, self.ni = int(fi), int(ni)
        self.fj, self.nj = int(fj), int(nj)
        self.k = float(k)
        self.L0 = float(L0)
        self.xtype = xtype
        # Set type-specific parameters
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
            # thin green/red segment
            self.line = gl.GLLinePlotItem(pos=np.zeros((2,3)), color=self.color, width=1, antialias=True)
            view.addItem(self.line)

    def key(self):
        # order-agnostic key to avoid duplicates
        a = (self.fi, self.ni); b = (self.fj, self.nj)
        return tuple(sorted((a, b)))

    def values(self, fibers):
        pi = fibers[self.fi].x[self.ni]
        pj = fibers[self.fj].x[self.nj]
        d  = pj - pi
        L  = float(np.linalg.norm(d))
        if L < 1e-12:
            u = np.array([0.0,0.0,0.0])
        else:
            u = d / L
        if self.xtype == 'covalent':
            # Covalent: allow tension and compression
            e = (L - self.L0)
        else:
            # Ionic/transient: tension-only (no compression)
            e = max(L - self.L0, 0.0)
        F  = self.k * e
        return pi, pj, u, L, e, F

    def apply_force(self, fibers):
        pi, pj, u, L, e, F = self.values(fibers)
        # Apply equal and opposite forces (allowing both tension and compression)
        fibers[self.fi].apply_external_force(self.ni, +F * u)
        fibers[self.fj].apply_external_force(self.nj, -F * u)
        return L, F

    def redraw(self, fibers):
        if self.line is None: return
        pi = fibers[self.fi].x[self.ni]
        pj = fibers[self.fj].x[self.nj]
        self.line.setData(pos=np.vstack([pi, pj]), color=self.color)

    def koff(self, F):
        """Bell slip-bond with numeric safety.
        - Covalent: self.koff0 == 0 -> never unbinds.
        - Ionic/transient: use tension-only force contribution.
        - Clamp force and exponent to avoid overflow.
        """
        if getattr(self, 'koff0', 0.0) == 0.0:
            return 0.0
        F_eff = max(0.0, float(F))
        F_eff = min(F_eff, KFF_FORCE_CLAMP_N)
        x = (F_eff * self.dx) / (KB_T + 1e-30)
        if x < 0.0: x = 0.0
        if x > KFF_EXP_CLAMP: x = KFF_EXP_CLAMP
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
        self.links = {}   # key -> Crosslink

    def _pair_iter(self, fibers):
        # Iterate distinct-fiber node pairs with stride
        for fi, f in enumerate(fibers):
            for fj in range(fi+1, len(fibers)):
                g = fibers[fj]
                for ni in range(1, f.N-1, CROSSLINK_STRIDE):
                    pi = f.x[ni]
                    # quick AABB filter (optional)
                    for nj in range(1, g.N-1, CROSSLINK_STRIDE):
                        pj = g.x[nj]
                        yield fi, ni, fj, nj, pi, pj

    def formation_step(self, fibers, dt):
        # Try to form new links near each other
        for fi, ni, fj, nj, pi, pj in self._pair_iter(fibers):
            d = float(np.linalg.norm(pj - pi))
            if d > CROSSLINK_RANGE:
                continue
            key = tuple(sorted(((fi,ni),(fj,nj))))
            if key in self.links:
                continue
            p_on = 1.0 - np.exp(-KFF_ON * dt)
            if random.random() < p_on:
                # stress-free at formation
                xtype = 'ionic' if random.random() < IONIC_FRACTION else 'covalent'
                xl = Crosslink(self.view, fi, ni, fj, nj, L0=d, k=CROSSLINK_K_SIM, xtype=xtype)
                self.links[xl.key()] = xl
                log_xl_event("bind", current_step, sim_time, fi, ni, fj, nj, d, d, 0.0, KFF_OFF0)

    def force_and_break_step(self, fibers, dt):
        # Apply forces from all links; then test unbinding with Bell model
        dead = []
        for key, xl in list(self.links.items()):
            L, F_sim = xl.apply_force(fibers)
            # Off-rate uses tension-only extension (meters) -> pN via KFF_PN_PER_NM
            e_m  = max(L - xl.L0, 0.0)
            e_nm = e_m * 1e9
            F_off_pN = e_nm * KFF_PN_PER_NM
            F_off_N  = F_off_pN * 1e-12
            koff = xl.koff(F_off_N)
            p_off = -np.expm1(-koff * dt)
            if np.random.random() < p_off:
                log_xl_event("unbind", current_step, sim_time, xl.fi, xl.ni, xl.fj, xl.nj, L, xl.L0, F_sim, koff)
                dead.append(key)
        for key in dead:
            self.links[key].remove_from_view(self.view)
            del self.links[key]


    def redraw(self, fibers):
        if not DRAW_CROSSLINKS:
            return
        for xl in self.links.values():
            xl.redraw(fibers)

# ==================== PyQt + Simulation ====================

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.setWindowTitle('Integrin Springs (Top Only) + Catch–Slip (hinged) + Fiber–Fiber Links')
view.setGeometry(0, 110, 1280, 800)
view.setCameraPosition(distance=2.0, elevation=20, azimuth=45)
view.show()

manager = IntegrinManager(view)
manager.populate_integrins(N=70, radius=0.1, z=0.0)  # plane z=0

fibers = []
for _ in range(25):
    p1 = np.random.uniform(low=[-0.1, -0.1, 0.004], high=[0.1, 0.1, 0.01])
    p2 = np.random.uniform(low=[-0.1, -0.1, 0.004], high=[0.1, 0.1, 0.01])
    fibers.append(ElasticFiberFEM(start_point=p1, end_point=p2))

crosslinks = CrosslinkManager(view)

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
    # Crosslink logs
    with open(XL_EVENTS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event","step","time_s","fiber_i","node_i","fiber_j","node_j","L","L0","F","koff"])
    with open(XL_SERIES_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step","time_s","fiber_i","node_i","fiber_j","node_j","L","L0","F","koff"])

    for f in fibers:
        f.line.setData(pos=f.x)

    manager.activate_near_fibers(fibers, threshold=None, save_path="integrin_distances3.csv")
    capture_frame()

def frame_3():
    # deformation log
    with open("fiber_deformation3.csv", "w", newline="") as fh:
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

        # --- Crosslink formation BEFORE forces ---
        crosslinks.formation_step(fibers, dt)

        if REACTIVATE:
            manager.activate_near_fibers(fibers, threshold=None, save_path=None)

        active = [i for i in manager.integrins.values() if i.state == 'active']
        print(f"Step {current_step}: active integrins = {len(active)}, fiber-fiber links = {len(crosslinks.links)}")

        # log BEFORE
        with open("fiber_deformation3.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            for j, f in enumerate(fibers):
                for i, node in enumerate(f.x):
                    w.writerow([f"STEP_{current_step}_BEFORE", j, i, *node])

        # --- Integrin forces and stochastic unbinding ---
        still_bound = []
        for integ in active:
            if integ.attachment is None:
                continue
            fib, idx = integ.attachment
            attach_pt = fib.x[idx]
            pivot = integ._pivot_point()

            # active contraction (rest-length change) of TOP spring only, with hard floor
            integ.L0 = max(integ.inactive_top_length, float(integ.L0) - float(integ.L0_shrink_rate) * dt)

            # TOP spring geometry: pivot -> attach
            d_vec_top = attach_pt - pivot
            L_top = float(np.linalg.norm(d_vec_top))
            if L_top < 1e-9:
                integ._draw_active_hinged_to(attach_pt)
                continue

            # ---- HARD CAP: if the fiber moved out of reach, unbind instead of overstretching ----
            if L_top > integ.top_max:  # <<< NEW
                fib_idx = fibers.index(fib)
                # log an overstretch unbind (reuse existing schema)
                e_tmp = max(L_top - integ.L0, 0.0)
                F_sim_tmp = SPRING_K_SIM * e_tmp
                F_phys_tmp = (K_PN_PER_NM * (e_tmp * 1e9)) * 1e-12
                koff_tmp = Integrin.catch_slip_off_rate(F_phys_tmp)
                log_bond_event("unbind", current_step, sim_time, integ.id, fib_idx, idx,
                               L_top, integ.L0, e_tmp, F_sim_tmp, F_phys_tmp, koff_tmp,
                               lifetime=integ.bound_time)
                integ.unbind()
                continue

            u_top = d_vec_top / L_top
            integ.length = (integ.inactive_length * 0.5) + min(L_top, integ.top_max)  # visual safety
            integ.bound_time += dt

            # extension and forces (TOP spring)
            e = max(L_top - integ.L0, 0.0)    # no compression; cannot pull below pivot+top-rest
            F_sim  = SPRING_K_SIM * e
            e_nm   = e * 1e9
            F_phys = (K_PN_PER_NM * e_nm) * 1e-12
            koff   = Integrin.catch_slip_off_rate(F_phys)

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
            integ._draw_active_hinged_to(attach_pt)  # respects clamp in draw
            if F_sim > 0:
                fib.apply_external_force(idx, -F_sim * u_top)

            still_bound.append((integ, fib, idx))

        # --- Crosslink forces and stochastic dissolution ---
        crosslinks.force_and_break_step(fibers, dt)

        # --- Update fibers once with all external forces (integrins + crosslinks) ---
        for f in fibers:
            f.update()

        # redraw crosslinks and integrins after update
        crosslinks.redraw(fibers)

        for integ, fib, idx in still_bound:
            attach_post = fib.x[idx]
            pivot = integ._pivot_point()

            d_top_post = attach_post - pivot
            L_top_post = float(np.linalg.norm(d_top_post))
            # keep the display in sync (draw clamps length)
            integ.remove_integrin()
            integ._draw_active_hinged_to(attach_post)

            if not UNBIND_BEFORE_FORCE:
                # if post-update moved out of reach, unbind
                if L_top_post > integ.top_max:  # <<< NEW (symmetric safety)
                    fib_idx = fibers.index(fib)
                    e_tmp = max(L_top_post - integ.L0, 0.0)
                    F_sim_tmp = SPRING_K_SIM * e_tmp
                    F_phys_tmp = (K_PN_PER_NM * (e_tmp * 1e9)) * 1e-12
                    koff_tmp = Integrin.catch_slip_off_rate(F_phys_tmp)
                    log_bond_event("unbind", current_step, sim_time, integ.id, fib_idx, idx,
                                   L_top_post, integ.L0, e_tmp, F_sim_tmp, F_phys_tmp, koff_tmp,
                                   lifetime=integ.bound_time)
                    integ.unbind()
                    continue

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
        with open("fiber_deformation3.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            for j, f in enumerate(fibers):
                for i, node in enumerate(f.x):
                    w.writerow([f"STEP_{current_step}_AFTER", j, i, *node])

        capture_frame()

        # advance time/step
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
    print("Saving video to integrin_simulation3.mp4...")
    imageio.mimsave("integrin_simulation3.mp4", frame_images, fps=2)

QtCore.QTimer.singleShot(1000, frame_1)
QtCore.QTimer.singleShot(2500, frame_2)
QtCore.QTimer.singleShot(4000, frame_3)

app.exec_()
