"""
Tests for placing CUT3R's world frame in the map, which need no ROS.

The scenario is built forwards -- a world frame put somewhere in the map, a
robot and a neck, and the camera pose CUT3R would report -- and the module is
asked to recover where the world frame was put.  A composition in the wrong
order, an inverse on the wrong factor or a camera block read in the wrong
convention all return a transform, just not that one.
"""

import math

import numpy as np
import pytest

from mecanumbot_deep3r import camera_pose, world_frame
from mecanumbot_deep3r.camera_pose import NeckCamera


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rigid(rot, t):
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = t
    return out


def a_frame(x=1.5, y=-0.7, yaw=0.9, ticks=520):
    """Return the header the node sends, and T_map_optical it describes."""
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    extrinsic = NeckCamera().extrinsic(ticks)
    header = camera_pose.frame_pose_header(
        (x, y, 0.0), q, frame="map", child_frame="mecanumbot/base_link",
        camera=extrinsic)
    return header, rigid(rot_z(yaw), (x, y, 0.0)) @ extrinsic


# Where the reconstruction's world frame was put: turned and tipped, since
# CUT3R's world is the first camera's optical frame and so is never level.
T_MAP_WORLD = rigid(rot_z(-2.1) @ rot_x(1.3), (4.0, 2.5, 0.2))


def cut3r_pose(t_map_optical):
    """Return the pose_c2w CUT3R reports for a camera at ``t_map_optical``."""
    return world_frame.invert_rigid(T_MAP_WORLD) @ t_map_optical


class TestRecoveringTheWorldFrame:

    def test_it_is_where_it_was_put(self):
        header, t_map_optical = a_frame()
        got = world_frame.map_from_world(header, cut3r_pose(t_map_optical))
        assert got == pytest.approx(T_MAP_WORLD, abs=1e-9)

    def test_it_stays_put_while_the_robot_drives_and_the_neck_moves(self):
        """The property RViz relies on: one world frame, whatever the camera did."""
        estimates = []
        for x, y, yaw, ticks in [(0, 0, 0, 600), (2, 1, 1.2, 300), (-1, 3, -2.5, 800)]:
            header, t_map_optical = a_frame(x, y, yaw, ticks)
            estimates.append(world_frame.map_from_world(header, cut3r_pose(t_map_optical)))
        for estimate in estimates[1:]:
            metres, radians = world_frame.spread(estimates[0], estimate)
            assert metres == pytest.approx(0.0, abs=1e-9)
            assert radians == pytest.approx(0.0, abs=1e-6)

    def test_a_cloud_point_lands_where_the_camera_saw_it(self):
        """One metre out of the lens, in the world frame, is one metre out of it in the map."""
        header, t_map_optical = a_frame()
        c2w = cut3r_pose(t_map_optical)
        in_world = c2w @ np.array([0.0, 0.0, 1.0, 1.0])
        in_map = world_frame.map_from_world(header, c2w) @ in_world
        assert in_map == pytest.approx(t_map_optical @ np.array([0.0, 0.0, 1.0, 1.0]))

    def test_the_base_is_planar_as_the_server_takes_it(self):
        """A quaternion with roll in it is ignored: the server composes the yaw only."""
        header, t_map_optical = a_frame()
        tipped = dict(header, qx=0.3, qw=math.sqrt(1 - 0.3 ** 2 - header["qz"] ** 2))
        got = world_frame.map_from_world(tipped, cut3r_pose(t_map_optical))
        assert got == pytest.approx(T_MAP_WORLD, abs=1e-9)


class TestWhenThereIsNothingToPlace:

    def test_a_frame_that_went_without_a_pose(self):
        assert world_frame.map_from_world(None, np.eye(4)) is None

    def test_a_frame_that_went_without_its_camera(self):
        """The server then used its own mount, which this node does not know."""
        header, _ = a_frame()
        del header["camera"]
        assert world_frame.map_from_world(header, np.eye(4)) is None

    def test_a_reply_without_a_cloud_pose(self):
        header, _ = a_frame()
        assert world_frame.map_from_world(header, None) is None

    def test_a_malformed_cloud_pose_is_refused_not_guessed(self):
        header, _ = a_frame()
        with pytest.raises(world_frame.PlacementError):
            world_frame.map_from_world(header, [[1.0, 0.0], [0.0, 1.0]])
        bad = np.eye(4)
        bad[0, 3] = float("nan")
        with pytest.raises(world_frame.PlacementError):
            world_frame.map_from_world(header, bad)


class TestTheParts:

    def test_transform_parts_round_trip_in_ros_order(self):
        (x, y, z), (qx, qy, qz, qw) = world_frame.transform_parts(T_MAP_WORLD)
        block = {"x": x, "y": y, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz}
        assert world_frame.matrix_from_pose(block) == pytest.approx(T_MAP_WORLD, abs=1e-9)

    def test_spread_measures_distance_and_angle(self):
        moved = rigid(rot_z(0.25), (0.3, 0.4, 0.0)) @ np.eye(4)
        metres, radians = world_frame.spread(np.eye(4), moved)
        assert metres == pytest.approx(0.5)
        assert radians == pytest.approx(0.25)
