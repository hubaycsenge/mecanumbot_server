"""
Tests for the neck camera model and the pose header, which need no ROS.

The failures worth catching are the ones that place a cloud plausibly and
wrongly: a tilt with the wrong sign, a lens that does not move with the head,
an optical frame with its axes swapped.
"""

import math

import numpy as np
import pytest

from mecanumbot_deep3r import camera_pose
from mecanumbot_deep3r.camera_pose import NeckCamera


def as_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_at_level_the_lens_looks_forward_with_the_image_upright():
    t = NeckCamera().extrinsic(600)
    # Optical z (out of the lens) is the robot's forward, optical y (down the
    # image) is the robot's down, optical x (along the image) is its right.
    assert t[:3, 2] == pytest.approx([1.0, 0.0, 0.0])
    assert t[:3, 1] == pytest.approx([0.0, 0.0, -1.0])
    assert t[:3, 0] == pytest.approx([0.0, -1.0, 0.0])


def test_at_level_the_lens_is_where_perception_measured_it():
    """0.13 / 0.21 in mecanumbot_sensorprocess_smart, reached independently."""
    t = NeckCamera().extrinsic(600)
    assert t[0, 3] == pytest.approx(0.13, abs=0.005)
    assert t[2, 3] == pytest.approx(0.21, abs=0.005)
    assert t[1, 3] == 0.0


def test_a_larger_neck_position_looks_up():
    """The trees' convention: 7.0 is the seek pose, above the 6.0 level gaze."""
    axis = NeckCamera().extrinsic(700)[:3, 2]
    assert axis[2] > 0.0
    assert math.degrees(math.atan2(axis[2], axis[0])) == pytest.approx(
        math.degrees(100 * 0.005061))


def test_a_smaller_neck_position_looks_down():
    assert NeckCamera().extrinsic(450)[:3, 2][2] < 0.0


def test_the_lens_turns_with_the_head_about_the_pivot():
    cam = NeckCamera()
    pivot = np.array([cam.pivot_x, 0.0, cam.pivot_z])
    lever = math.hypot(cam.lever_x, cam.lever_z)
    for ticks in (300, 600, 800):
        lens = cam.extrinsic(ticks)[:3, 3]
        assert np.linalg.norm(lens - pivot) == pytest.approx(lever)
    # And it does move: a camera fixed at the level position would not.
    assert cam.extrinsic(800)[:3, 3] != pytest.approx(cam.extrinsic(600)[:3, 3])


def test_every_extrinsic_is_a_rigid_transform():
    for ticks in (200, 450, 600, 860):
        rot = NeckCamera().extrinsic(ticks)[:3, :3]
        assert rot @ rot.T == pytest.approx(np.eye(3))
        assert np.linalg.det(rot) == pytest.approx(1.0)


def test_the_measured_offset_at_level_is_applied():
    level = NeckCamera(pitch_at_level=math.radians(-5.0)).extrinsic(600)[:3, 2]
    assert math.degrees(math.atan2(level[2], level[0])) == pytest.approx(-5.0)


def test_an_unset_board_is_not_a_head_position():
    """The board reports 0 before its first command: -174 degrees."""
    assert NeckCamera().extrinsic(0) is None
    assert NeckCamera().extrinsic(float("nan")) is None


def test_the_header_carries_the_base_pose_in_ros_order():
    half = 0.25
    header = camera_pose.frame_pose_header(
        (1.0, 2.0, 0.01), (0.0, 0.0, math.sin(half), math.cos(half)),
        frame="map", child_frame="mecanumbot/base_link", age_ms=3.14159)
    assert header["frame"] == "map"
    assert header["child_frame"] == "mecanumbot/base_link"
    assert (header["x"], header["y"], header["z"]) == (1.0, 2.0, 0.01)
    assert header["yaw"] == pytest.approx(0.5)
    assert header["qw"] == pytest.approx(math.cos(half))
    assert header["qz"] == pytest.approx(math.sin(half))
    assert header["age_ms"] == 3.14
    assert "camera" not in header


def test_the_camera_block_round_trips_to_the_extrinsic():
    t = NeckCamera().extrinsic(520)
    header = camera_pose.frame_pose_header(
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
        frame="map", child_frame="mecanumbot/base_link", camera=t,
        camera_info={"source": "neck_model", "neck_ticks": 520})
    block = header["camera"]
    # Relative to the base the pose names -- the server refuses anything else.
    assert block["frame"] == "mecanumbot/base_link"
    assert block["source"] == "neck_model" and block["neck_ticks"] == 520
    rot = as_matrix((block["qw"], block["qx"], block["qy"], block["qz"]))
    assert rot == pytest.approx(t[:3, :3], abs=1e-9)
    assert [block["x"], block["y"], block["z"]] == pytest.approx(t[:3, 3])


def test_optical_to_body_matches_the_servers():
    """Written out twice across two repositories; checked against its meaning."""
    m = camera_pose.OPTICAL_TO_BODY
    assert m @ [0, 0, 1] == pytest.approx([1, 0, 0])     # lens -> forward
    assert m @ [1, 0, 0] == pytest.approx([0, -1, 0])    # image right -> right
    assert m @ [0, 1, 0] == pytest.approx([0, 0, -1])    # image down -> down
