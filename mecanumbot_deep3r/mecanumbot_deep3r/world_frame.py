"""
Where CUT3R's world frame sits in the robot's map, for one frame.

The cloud comes back in CUT3R's own world frame, anchored on the first frame of
each reconstruction session with an origin and an orientation nobody chose.
What relates it to the map is the camera, whose pose is known in both::

    T_map_world = T_map_base · T_base_optical · (T_world_optical)⁻¹

``T_world_optical`` is the ``pose_c2w`` the server returns with the cloud.  The
other two are exactly what this node attached to the frame it sent -- the base
at the image's stamp and the camera from the neck -- so the composition needs
nothing from the server beyond the cloud's own pose.

It is the server's composition, not a second opinion
----------------------------------------------------
``map_from_cloud_matrix`` in RoboCamStreamProcessing's ``robocam/compare.py``
is the same product over the same inputs, and it is what places the cloud for
the comparison.  Two details are copied from it on purpose, so that a cloud
shown in RViz through this transform lands where the comparison put it:

* The base is **planar**: ``x, y, z`` and yaw, with roll and pitch taken as
  zero.  The quaternion also travels, but the server does not use it.
* A frame the server placed with anything else -- the 10 Hz stream pose, or
  its configured camera mount -- has no transform here, because this node does
  not know what those were.  The result's ``pose_source`` says which it was.

Nothing is smoothed.  The estimate differs from frame to frame by odometry
error, the neck model's error and CUT3R's pose drift, and averaging it would
place each cloud somewhere the comparison did not.  ``spread`` measures that
difference instead, because a world frame that wanders in the map is the same
fault as an ``agreement`` near zero, seen from the other side.

Kept free of ROS, like ``geometry`` and ``camera_pose``.
"""

import math

import numpy as np

from .geometry import quat_from_matrix


class PlacementError(ValueError):
    """The inputs do not describe a placement the server would have made."""


def planar_base(x, y, z, yaw):
    """Return the 4x4 ``T_map_base`` for a robot on a floor."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = (x, y, z)
    return out


def invert_rigid(matrix):
    """Return the inverse of a 4x4 rigid transform, by transpose and negate."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rot, t = matrix[:3, :3], matrix[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ t
    return out


def matrix_from_pose(block):
    """Return the 4x4 of a pose block's ``x y z`` and ``qw qx qy qz``."""
    q = np.array([block["qw"], block["qx"], block["qy"], block["qz"]], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-9:
        raise PlacementError(f"quaternion {q.tolist()} is not a rotation")
    w, x, y, z = q / norm
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = (block["x"], block["y"], block["z"])
    return out


def map_from_world(frame_pose, pose_c2w):
    """
    Return ``T_map_world`` for one frame, or None when it cannot be had.

    ``frame_pose`` is the header ``camera_pose.frame_pose_header`` built for the
    frame; ``pose_c2w`` is the 4x4 the server returned with its cloud.  None
    when the frame went without a pose, or without a camera -- in both cases
    the server placed the cloud with something this node never saw.
    """
    if not frame_pose or frame_pose.get("camera") is None or pose_c2w is None:
        return None
    c2w = np.asarray(pose_c2w, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
        raise PlacementError(f"pose_c2w is not a finite 4x4: shape {c2w.shape}")
    t_map_base = planar_base(frame_pose["x"], frame_pose["y"], frame_pose["z"],
                             frame_pose["yaw"])
    t_base_optical = matrix_from_pose(frame_pose["camera"])
    return t_map_base @ t_base_optical @ invert_rigid(c2w)


def transform_parts(matrix):
    """Return ``((x, y, z), (qx, qy, qz, qw))`` in ROS order."""
    matrix = np.asarray(matrix, dtype=np.float64)
    qw, qx, qy, qz = quat_from_matrix(matrix[:3, :3])
    t = matrix[:3, 3]
    return (float(t[0]), float(t[1]), float(t[2])), (qx, qy, qz, qw)


def spread(reference, matrix):
    """
    Return ``(metres, radians)`` between two estimates of the same world frame.

    The distance between their origins and the angle of the rotation that
    takes one onto the other.  Zero for a reconstruction whose world frame
    stays put in the map, which is what a correct placement produces.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(matrix, dtype=np.float64)
    metres = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    cos = (float(np.trace(a[:3, :3].T @ b[:3, :3])) - 1.0) / 2.0
    return metres, math.acos(max(-1.0, min(1.0, cos)))
