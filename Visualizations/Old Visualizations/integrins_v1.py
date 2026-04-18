import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore


class Integrin:
    def __init__(self, view, id, base_position, length=0.1, radius=0.005, colors=[(1, 0, 0, 1), (0, 0, 1, 1)]):
        self.view = view
        self.base_position = base_position
        self.length = length
        self.radius = radius
        self.colors = colors
        self.state = 'inactive'
        self.id = id
        self.items = []

    def add_inactive_integrin(self):
        x, y, z = self.base_position
        self.items.clear()

        # Vertical segments (base)
        for i, color in enumerate(self.colors):
            cyl = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded', drawEdges=False)
            cyl.scale(self.radius, self.radius, self.length / 2)
            cyl.translate(x + (i - 0.5) * self.radius * 3, y, z)
            self.items.append(cyl)

        # Top bent segments
        for i, color in enumerate(self.colors):
            top = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=10, cols=20),
                                smooth=True, color=color, shader='shaded', drawEdges=False)
            top.rotate(90, 1, 0, 0)
            top.scale(self.radius, self.radius, self.length / 2)
            top.translate(x + (i - 0.5) * self.radius * 3, y, z + self.length / 2)
            self.items.append(top)

        self.view_integrin()

    def add_active_integrin(self):
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

    def switch_to_inactive(self):
        if self.state == 'active':
            self.remove_integrin()
            self.add_inactive_integrin()
            self.state = 'inactive'


class IntegrinManager:
    def __init__(self, view):
        self.view = view
        self.integrins = {}

    def populate_integrins(self, N, radius, z=0.0):
        positions = self._fibonacci_lattice_positions(N, radius, z)
        for i, pos in enumerate(positions):
            integrin = Integrin(self.view, id=f"integrin{i+1}", base_position=pos)
            integrin.add_inactive_integrin()
            self.integrins[integrin.id] = integrin

    def _fibonacci_lattice_positions(self, N, R, z):
        golden_angle = 2.3999632297
        positions = []
        for i in range(N):
            r = R * np.sqrt(i / N)
            theta = i * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append((x, y, z))
        return positions

    def toggle_states(self):
        for integrin in self.integrins.values():
            if integrin.state == 'inactive':
                integrin.switch_to_active()
            else:
                integrin.switch_to_inactive()


# Launch PyQt App
app = QtWidgets.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 2
w.setWindowTitle('Integrin Animation (L ↔ Line)')
w.setGeometry(0, 0, 800, 600)
w.show()

# Add grid
g = gl.GLGridItem()
g.scale(0.1, 0.1, 0.1)
w.addItem(g)

# Add integrins
manager = IntegrinManager(w)
manager.populate_integrins(N=6, radius=0.3)

# Set up timer to toggle active/inactive
timer = QtCore.QTimer()
timer.timeout.connect(manager.toggle_states)
timer.start(1500)  # milliseconds

# Run
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
