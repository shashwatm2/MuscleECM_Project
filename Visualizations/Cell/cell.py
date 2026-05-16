import sys
import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore, QtGui

# ==================== Overlay labels (2D QLabel projected from 3D) ====================

LABEL_FONT_PT    = 4.0
LABEL_COLOR_CSS = "rgba(120,120,120,0.98)"
LABEL_PAD        = 1

class OverlayLabeler(QtCore.QObject):
    """Projects 3D points to 2D and shows tiny QLabel numbers over the GL view."""
    def __init__(self, view: gl.GLViewWidget):
        super().__init__(parent=view)
        self.view = view
        self.labels = {}
        self.world_pos = {}
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
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
        if pos is None or lbl is None:
            return
        proj = self.view.projectionMatrix()
        viewm = self.view.viewMatrix()
        mvp = QtGui.QMatrix4x4(proj); mvp *= viewm
        v4 = QtGui.QVector4D(float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
        p4 = mvp.map(v4); w = p4.w()
        if w == 0:
            lbl.hide()
            return
        x_ndc, y_ndc, z_ndc = p4.x()/w, p4.y()/w, p4.z()/w
        if z_ndc < -1.0 or z_ndc > 1.0 or abs(x_ndc) > 1.2 or abs(y_ndc) > 1.2:
            lbl.hide()
            return
        vw, vh = max(1, self.view.width()), max(1, self.view.height())
        x_px = int((x_ndc * 0.5 + 0.5) * vw - lbl.width()/2)
        y_px = int(((1.0 - (y_ndc * 0.5 + 0.5)) * vh) - lbl.height()/2)
        lbl.move(x_px, y_px)
        lbl.show()

# ==================== Cell placement & scale ====================

CELL_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
CELL_RADIUS = 0.04

CLUSTERS_PER_CELL     = 10
INTEGRINS_PER_CLUSTER = 10
NUM_INTEGRINS = CLUSTERS_PER_CELL * INTEGRINS_PER_CLUSTER

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
    def __init__(self, view, id, base_position, length=0.03*0.5, radius=0.0025,
                 colors=[(1, 0, 0, 1), (0, 0, 1, 1)], cell_center=CELL_CENTER,
                 label_num=None, cluster_id=None):
        self.view = view
        self.id = id
        self.base_position = np.array(base_position, dtype=np.float64)
        self.center = np.array(cell_center, dtype=np.float64)
        self.length = float(length)
        self.radius = float(radius)
        self.colors = colors
        self.cluster_id = cluster_id
        self.state = 'inactive'
        self.items = []
        self.inactive_length = float(length)
        self.inactive_top_length = self.inactive_length * 0.5
        self.active_length   = 2.0 * float(length)
        self.top_max = self.inactive_length
        self.attachment = None
        self.bound_time = 0.0

        self.n = _unit(self.base_position - self.center)
        ref = np.array([0.0, 0.0, 1.0]) if abs(self.n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        self.t1 = _unit(np.cross(ref, self.n))
        self.t2 = _unit(np.cross(self.n, self.t1))

        self.label_num = label_num
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

        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded')
            self._add_item(
                cyl,
                translate=tuple((base + offsets[i]).astype(float)),
                rotate=rot_to_n,
                scale=(self.radius, self.radius, self.inactive_length)
            )

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

class IntegrinManager:
    def __init__(self, view):
        self.view = view
        self.integrins = {}
        self.counter = 0

    def populate_integrin_clusters_on_cell(
        self,
        center,
        radius,
        clusters_per_cell=CLUSTERS_PER_CELL,
        integrins_per_cluster=INTEGRINS_PER_CLUSTER,
    ):
        center = np.array(center, dtype=np.float64)
        cluster_centers = fibonacci_sphere_points(clusters_per_cell, radius, center)

        for cid, c_pt in enumerate(cluster_centers, start=1):
            n = _unit(c_pt - center)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            t1 = _unit(np.cross(ref, n))
            t2 = _unit(np.cross(n, t1))
            cluster_radius = 0.2 * radius

            for k in range(integrins_per_cluster):
                angle = 2.0 * np.pi * k / integrins_per_cluster
                offset = cluster_radius * (np.cos(angle) * t1 + np.sin(angle) * t2)
                base_pos = center + radius * n + offset

                self.counter += 1
                integ = Integrin(
                    self.view,
                    f"integrin{self.counter}",
                    base_pos,
                    cell_center=center,
                    label_num=self.counter,
                    cluster_id=cid,
                )
                self.integrins[integ.id] = integ

# ==================== PyQt cell-only viewer ====================

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
view = gl.GLViewWidget()
view.setWindowTitle('Cell Only')
view.setGeometry(0, 110, 1280, 800)
view.setBackgroundColor('w')
view.setCameraPosition(distance=0.18, elevation=20, azimuth=45)
view.show()

overlay = OverlayLabeler(view)

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

add_cell_mesh(CELL_CENTER)

manager = IntegrinManager(view)
manager.populate_integrin_clusters_on_cell(center=CELL_CENTER, radius=CELL_RADIUS)

print(f"Cell-only view loaded: {len(manager.integrins)} integrins")
sys.exit(app.exec_())
