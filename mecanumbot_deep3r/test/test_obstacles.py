"""
Tests for the server's obstacle cells, with no ROS and no server.

The thing these are really about is that slam_toolbox re-rasterises ``/map`` on
nearly every update during T1 -- new width, new height, new origin -- so a cell
accumulated as a grid index names a different place a few seconds later. An
off-by-one is invisible in RViz and is about the width of the robot's margin,
so it has to be checked here rather than noticed there.
"""

import numpy as np
import pytest

from mecanumbot_deep3r import obstacles as O


def patch(values, *, x0=0, y0=0, resolution=0.05, origin=(0.0, 0.0)):
    header = {"resolution": resolution, "origin": list(origin), "x0": x0, "y0": y0}
    return header, np.asarray(values, dtype=np.int8)


class TestAccumulating:

    def test_occupied_cells_are_kept(self):
        acc = O.CloudObstacles()
        assert acc.add(*patch([[100, 100], [100, 100]])) == 4
        assert len(acc) == 4

    def test_free_and_unknown_cells_add_nothing(self):
        """The server may add obstacles and never clear them; so may this."""
        acc = O.CloudObstacles()
        assert acc.add(*patch([[0, -1], [10, 49]])) == 0
        assert len(acc) == 0

    def test_the_same_cell_twice_is_one_cell(self):
        acc = O.CloudObstacles()
        acc.add(*patch([[100]], x0=3, y0=4))
        assert acc.add(*patch([[100]], x0=3, y0=4)) == 0
        assert len(acc) == 1

    def test_occupancy_is_raised_never_lowered(self):
        acc = O.CloudObstacles()
        acc.add(*patch([[100]]))
        acc.add(*patch([[60]]))
        grid = acc.render(0.05, 0.0, 0.0, 1, 1)
        assert grid[0, 0] == 100

    def test_a_patch_without_a_resolution_is_refused(self):
        acc = O.CloudObstacles()
        with pytest.raises(ValueError):
            acc.add({"resolution": 0.0, "origin": [0.0, 0.0]}, np.array([[100]], np.int8))

    def test_the_cap_stops_growth_and_says_so(self):
        acc = O.CloudObstacles(max_cells=3)
        acc.add(*patch(np.full((4, 4), 100, np.int8)))
        assert len(acc) == 3
        assert acc.capped is True

    def test_a_slam_restart_forgets_everything(self):
        """A coordinate in a map that no longer exists is not a place."""
        acc = O.CloudObstacles()
        acc.add(*patch([[100]]))
        acc.reset("map@0.0500#1")
        assert len(acc) == 0
        assert acc.map_id == "map@0.0500#1"


class TestRendering:

    def test_a_cell_lands_where_the_patch_put_it(self):
        acc = O.CloudObstacles()
        acc.add(*patch([[100]], x0=7, y0=11, origin=(1.0, 2.0)))
        grid = acc.render(0.05, 1.0, 2.0, 20, 20)
        assert np.argwhere(grid >= 0).tolist() == [[11, 7]]

    def test_a_regrown_map_does_not_move_the_obstacle(self):
        """
        The regression this class exists for.

        slam_toolbox sizes the grid to the bounding box of every scan so far,
        so seeing further left moves the origin and renumbers every column. An
        obstacle held in metres stays over the same floor; one held as an index
        would slide with the origin.
        """
        acc = O.CloudObstacles()
        acc.add(*patch([[100]], x0=7, y0=11, origin=(1.0, 2.0)))

        before = acc.render(0.05, 1.0, 2.0, 20, 20)
        # The map grows 10 cells left and 4 down; the same floor, renumbered.
        after = acc.render(0.05, 1.0 - 10 * 0.05, 2.0 - 4 * 0.05, 30, 30)

        assert np.argwhere(before >= 0).tolist() == [[11, 7]]
        assert np.argwhere(after >= 0).tolist() == [[15, 17]]

    def test_everything_unsaid_is_unknown(self):
        """A costmap layer reading this must add obstacles and clear nothing."""
        acc = O.CloudObstacles()
        acc.add(*patch([[100]]))
        grid = acc.render(0.05, 0.0, 0.0, 4, 4)
        assert (grid[grid != 100] == O.UNKNOWN).all()

    def test_cells_off_the_grid_are_dropped_not_clamped(self):
        """A clamped cell is an obstacle along the map edge, which blocks it."""
        acc = O.CloudObstacles()
        acc.add(*patch([[100]], x0=500, y0=500))
        grid = acc.render(0.05, 0.0, 0.0, 10, 10)
        assert (grid == O.UNKNOWN).all()

    def test_an_empty_accumulator_renders_an_unknown_grid(self):
        grid = O.CloudObstacles().render(0.05, 0.0, 0.0, 3, 2)
        assert grid.shape == (2, 3)
        assert (grid == O.UNKNOWN).all()
