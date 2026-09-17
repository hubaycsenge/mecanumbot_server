"""
The obstacles the server found, kept where a resizing SLAM map cannot move them.

``map_update`` is the server's other arrow back at the robot: a patch of cells
its reconstruction says are occupied and the scanner's one horizontal plane
never saw -- a table top, a low shelf, the lip of a step. Until 2026-09-17 the
client decoded every one of them and the node dropped them, so nothing on the
robot acted on the finest-grained thing the server produces.

This is what accumulates them. It is **navigation's** copy, not SLAM's:
slam_toolbox owns ``/map`` and builds it from its scan graph, and there is no
way to inject foreign cells into that which would still be a scan product. So
the cells go to a grid of their own, and a costmap layer reads it alongside the
map rather than instead of it.

Cells are held **in metres, not in grid indices**, and that is the whole design.
slam_toolbox re-rasterises to the bounding box of every scan so far, so during
T1 the grid's width, height and origin change on almost every ``/map`` update
(see ``bridge.MapIdentity`` for what keying on that geometry broke). An index
accumulated against one of those grids names a different place in the next one.
A metre coordinate names the same place in all of them.

Free of ROS, like ``bridge`` and ``geometry``, so the indexing can be tested
without a graph: an off-by-one here is an obstacle one cell from where it
really is, which is not visible in RViz and is exactly the width of a robot's
margin.
"""

from __future__ import annotations

import math

import numpy as np

#: ``nav_msgs/OccupancyGrid``'s "no information", and the server's ``unknown``.
UNKNOWN = -1

#: Occupancy at or above which a patch cell is an obstacle worth keeping. The
#: server sends the robot's own 0..100 scale; anything below this is either
#: free space or a cell it has no opinion about, and neither adds an obstacle.
#: Clearing is deliberately impossible -- see ``merge_map_patch``'s reasoning
#: about a monocular reconstruction being safe to be wrong in one direction
#: only.
OCCUPIED = 50


class CloudObstacles:
    """
    Every occupied cell the server has reported, in map-frame metres.

    ``max_cells`` bounds a long T1: at 5 cm a hundred thousand cells is a
    250 m^2 of solid obstacle, far more than a session should ever produce, so
    reaching it means something is wrong and the cap says so rather than
    growing until the Orin runs out of memory.
    """

    def __init__(self, max_cells=100000):
        self.max_cells = int(max_cells)
        self.map_id = ""
        self.capped = False
        #: (cell_x, cell_y) -> (x, y, occupancy). Keyed on a resolution-sized
        #: lattice so the same place reported twice is one cell.
        self._cells = {}

    def __len__(self):
        return len(self._cells)

    def reset(self, map_id=""):
        """
        Forget everything; the map these coordinates named is gone.

        Called on a SLAM restart. A point at (3.2, 1.4) in the old map is not
        the same place in the new one, and keeping it would put an obstacle
        somewhere nothing was ever seen.
        """
        self._cells.clear()
        self.capped = False
        self.map_id = str(map_id or "")

    def add(self, header, cells):
        """
        Merge one decoded ``map_update`` patch; return how many cells it added.

        ``header`` carries the resolution and origin of the grid the patch
        indexes, so the conversion needs nothing from whatever ``/map`` looks
        like now -- which is the point, since it will have changed.
        """
        if cells is None:
            return 0
        cells = np.asarray(cells)
        if cells.ndim != 2:
            raise ValueError(f"a map patch is 2D, got shape {cells.shape}")

        resolution = float(header.get("resolution", 0.0))
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError(f"a patch needs a positive resolution, got {resolution!r}")
        origin = header.get("origin") or (0.0, 0.0)
        origin_x, origin_y = float(origin[0]), float(origin[1])
        x0 = int(header.get("x0", 0))
        y0 = int(header.get("y0", 0))

        rows, cols = np.nonzero(cells >= OCCUPIED)
        added = 0
        for row, col in zip(rows.tolist(), cols.tolist()):
            # The cell's centre, which is what survives a re-rasterisation.
            x = origin_x + (x0 + col + 0.5) * resolution
            y = origin_y + (y0 + row + 0.5) * resolution
            key = (int(math.floor(x / resolution)), int(math.floor(y / resolution)))
            value = int(cells[row, col])
            known = self._cells.get(key)
            if known is None:
                if len(self._cells) >= self.max_cells:
                    self.capped = True
                    break
                self._cells[key] = (x, y, value)
                added += 1
            elif value > known[2]:
                # Raise only, never lower: the server's own merge rule, for the
                # same reason -- an invented obstacle costs a detour, a cleared
                # real one costs a collision.
                self._cells[key] = (known[0], known[1], value)
        return added

    def render(self, resolution, origin_x, origin_y, width, height):
        """
        Draw the accumulated cells into a grid of the given geometry.

        Returns an ``int8`` (height, width) array that is ``UNKNOWN`` wherever
        the server has said nothing, so a costmap layer reading it adds
        obstacles and clears nothing.  Cells outside the grid are dropped
        rather than clamped: a clamped cell is an obstacle against the map's
        edge, which is a place the robot would then refuse to go.
        """
        width, height = int(width), int(height)
        grid = np.full((height, width), UNKNOWN, dtype=np.int8)
        if width <= 0 or height <= 0 or not self._cells:
            return grid
        resolution = float(resolution)
        if resolution <= 0.0:
            raise ValueError(f"a grid needs a positive resolution, got {resolution!r}")

        points = np.fromiter(
            (v for cell in self._cells.values() for v in cell),
            dtype=np.float64, count=3 * len(self._cells),
        ).reshape(-1, 3)
        cols = np.floor((points[:, 0] - float(origin_x)) / resolution).astype(np.int64)
        rows = np.floor((points[:, 1] - float(origin_y)) / resolution).astype(np.int64)
        inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        grid[rows[inside], cols[inside]] = points[inside, 2].astype(np.int8)
        return grid
