"""
Grid of read-only canvases showing the results of several models side by side.
"""

import math

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QWidget

from .canvas import Canvas


def grid_shape(count):
    """(rows, columns) of a grid holding count panels, as close to square as fits.

    The columns are the rounded-up square root of count, so a grid is never
    taller than it is wide: 2 -> 1x2, 3 -> 2x2, 6 -> 2x3, 9 -> 3x3.
    """
    if count <= 0:
        return 0, 0
    columns = math.ceil(math.sqrt(count))
    return math.ceil(count / columns), columns


def grid_cells(count):
    """(row, half-column) of each of count panels, in order.

    Each panel spans two half-columns, so a last row with fewer panels than
    columns can be shifted by half a panel and sit centred under the rows above.
    """
    rows, columns = grid_shape(count)
    cells = []
    for index in range(count):
        row, column = divmod(index, columns)
        in_row = columns if row < rows - 1 else count - row * columns
        cells.append((row, 2 * column + (columns - in_row)))
    return cells


class CompareGrid(QWidget):
    navigation_requested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._panels = {}
        self._with_result = set()

    def models(self):
        return list(self._panels)

    def set_models(self, names):
        """Show one panel per model, laid out by grid_cells."""
        for panel in self._panels.values():
            self._layout.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self._panels = {}
        self._with_result.clear()
        for stretch_index in range(self._layout.rowCount()):
            self._layout.setRowStretch(stretch_index, 0)
        for stretch_index in range(self._layout.columnCount()):
            self._layout.setColumnStretch(stretch_index, 0)
        rows, columns = grid_shape(len(names))
        for name, (row, half_column) in zip(names, grid_cells(len(names))):
            panel = Canvas()
            panel.set_read_only(True)
            panel.set_title(name)
            panel.view_changed.connect(self._share_view)
            panel.navigation_requested.connect(self.navigation_requested)
            self._layout.addWidget(panel, row, half_column, 1, 2)
            panel.show()
            self._panels[name] = panel
        for row in range(rows):
            self._layout.setRowStretch(row, 1)
        for half_column in range(2 * columns):
            self._layout.setColumnStretch(half_column, 1)
        # The panels fit the image to their size when it is loaded, so they
        # take their place in the grid first.
        self._layout.activate()

    def set_image(self, image, roi_box, opacity, contour_only):
        """Show the same image, ROI and mask style in every panel.

        An image already shown is not loaded again, so the panels keep their zoom.
        """
        for panel in self._panels.values():
            if panel.image is not image:
                panel.load_image_array(image)
            panel.set_roi(roi_box)
            panel.set_mask_opacity(opacity)
            panel.set_contour_only(contour_only)

    def set_regions(self, region_view, knee_side, centre_x):
        """Split every panel's mask into regions the way the main canvas does."""
        for panel in self._panels.values():
            panel.set_knee_side(knee_side)
            panel.set_centre_x(centre_x)
            panel.set_region_view(region_view)

    def visible_mask(self, name):
        """A model's mask as shown, empty outside the ROI, or None without one."""
        panel = self._panels.get(name)
        if panel is None or panel.image is None or name not in self._with_result:
            return None
        return panel.visible_mask()

    def clear(self):
        self._with_result.clear()
        for panel in self._panels.values():
            panel.clear_image()

    def set_mask(self, name, mask):
        """Show a model's mask, or mark its panel as having no saved result."""
        panel = self._panels.get(name)
        if panel is None or panel.image is None:
            return
        if mask is None:
            panel.clear_mask()
            panel.set_title(f"{name} (no saved result)")
            self._with_result.discard(name)
        else:
            panel.set_mask(mask)
            panel.set_title(name)
            self._with_result.add(name)

    def fit_to_window(self):
        for panel in self._panels.values():
            panel.fit_to_window()

    def _share_view(self, scale, scroll_x, scroll_y):
        source = self.sender()
        for panel in self._panels.values():
            if panel is not source:
                panel.set_view(scale, scroll_x, scroll_y)
