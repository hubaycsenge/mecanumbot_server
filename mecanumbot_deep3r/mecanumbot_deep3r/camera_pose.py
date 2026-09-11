"""
Where the camera was for one frame, and the pose header that carries it.

The Mecanumbot's camera sits on a neck servo, so where the camera is changes
whenever a tree tilts the head -- and the fetch tree sweeps it continuously.
The server used to place every cloud with one fixed mount from its own config,
so a frame taken with the head tilted down was placed as if it looked level:
everything in it rotated about the lens by the tilt.

Not read from TF, deliberately
------------------------------
The URDF's ``head_link`` is rotated 90 degrees about x and ``camera_link``
another 90 about y, for the meshes. Composed, ``camera_rgb_optical_frame``
looks along the neck's own rotation axis -- to the robot's right -- at every
neck angle, so in TF the neck spins the image instead of tilting it.
``mecanumbot_sensorprocess_smart`` met the same wall and took its camera height
and tilt from measured parameters instead; this does the same, with the neck
added. ``head_joint`` in ``joint_states`` is no way round it either: its zero is
uncalibrated, and the simulator converts servo ticks with the opposite sign.

The model
---------
A pivot fixed on the base, a lever from the pivot to the lens that turns with
the head, and a tilt that is linear in the servo's ticks::

    pitch          = pitch_at_level + (ticks - level_ticks) * rad_per_tick
    lens           = pivot + R(pitch) @ lever
    T_base_optical = [R(pitch) @ OPTICAL_TO_BODY | lens]

Pitch is positive **up**, matching the trees' "larger neck position looks
further up". The pivot and lever are the URDF's translations -- the one part of
it written for the robot rather than the mesh -- and at level they put the lens
at (0.128, 0.206) m in ``base_link``, agreeing with the 0.13 / 0.21 perception
uses. ``level_ticks`` is the trees' ``neck_level_pos`` (6.0 board units, the
"neutral driving gaze"), and ``rad_per_tick`` is the constant
``mecanumbot_sensorproc_node`` uses for the same servo. Whether that gaze is
*optically* level has not been measured; ``pitch_at_level`` is the number to
set when it is.

``ticks`` is the neck's **goal**: the firmware echoes the last command back as
``opencr_state.pos_n`` and never reads the AX-12A's present position, so during
a sweep this model is ahead of the head by the servo's travel time.

Kept free of ROS, like ``geometry``, so it can be tested on its own.
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import quat_from_matrix, yaw_from_quaternion

#: Vision optical (x right, y down, z forward) -> body (x forward, y left, z up).
#: The same matrix as ``OPTICAL_TO_BODY`` in RoboCamStreamProcessing's
#: ``robocam/compare.py``; the server composes what this module sends onto it.
OPTICAL_TO_BODY = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

#: The optical frame's name in the header. Not a TF frame -- see above.
CAMERA_FRAME = "camera_optical"


@dataclass(frozen=True)
class NeckCamera:
    """The camera on the neck, in ``base_link`` (metres, radians)."""

    pivot_x: float = 0.1063
    pivot_z: float = 0.1679
    lever_x: float = 0.022
    lever_z: float = 0.038
    level_ticks: float = 600.0
    rad_per_tick: float = 0.005061
    pitch_at_level: float = 0.0
    # Beyond this a reading is not a head position. The board reports 0 before
    # its first command, which is -174 degrees here, and that is the case this
    # is for; the neck's own 200..860 range stays inside it.
    max_abs_pitch: float = math.radians(120.0)

    def pitch(self, ticks):
        """Return the camera's pitch for a neck at ``ticks``, positive up."""
        return self.pitch_at_level + (float(ticks) - self.level_ticks) * self.rad_per_tick

    def extrinsic(self, ticks):
        """
        Return the 4x4 ``T_base_optical`` for a neck at ``ticks``.

        None when the reading implies a pitch the head cannot have: sending the
        server a camera looking backwards would place the cloud there with full
        confidence.
        """
        pitch = self.pitch(ticks)
        if not math.isfinite(pitch) or abs(pitch) > self.max_abs_pitch:
            return None
        c, s = math.cos(pitch), math.sin(pitch)
        # About the base's y axis by -pitch: a positive rotation about y tips x
        # down under REP-103, and pitch here is positive up.
        tilt = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        out = np.eye(4)
        out[:3, :3] = tilt @ OPTICAL_TO_BODY
        out[:3, 3] = (np.array([self.pivot_x, 0.0, self.pivot_z])
                      + tilt @ np.array([self.lever_x, 0.0, self.lever_z]))
        return out

    def describe(self):
        """Return one log line saying what this model assumes."""
        level = self.extrinsic(self.level_ticks)
        low = math.degrees(self.pitch(200))
        high = math.degrees(self.pitch(860))
        return (
            f"camera pose from the neck: lens at ({level[0, 3]:.3f}, {level[2, 3]:.3f}) m "
            f"in base_link at {self.level_ticks:.0f} ticks, pitch "
            f"{math.degrees(self.pitch_at_level):+.1f} deg there, "
            f"{math.degrees(self.rad_per_tick):.3f} deg per tick "
            f"({low:+.0f}..{high:+.0f} deg over the neck's 200..860)"
        )


def frame_pose_header(translation, rotation_xyzw, *, frame, child_frame,
                      age_ms=0.0, camera=None, camera_info=None):
    """
    Build the ``pose`` a frame header carries.

    The fields are an ``odom`` header's, so the server decodes it with the same
    checks as a streamed pose; ``camera``, when given, is ``T_base_optical`` and
    goes out as a quaternion and a translation relative to ``child_frame``.
    ``rotation_xyzw`` is in ROS order, as a ``geometry_msgs`` quaternion holds it.
    """
    x, y, z = (float(v) for v in translation)
    qx, qy, qz, qw = (float(v) for v in rotation_xyzw)
    pose = {
        "frame": str(frame),
        "child_frame": str(child_frame),
        "x": x, "y": y, "z": z,
        "yaw": yaw_from_quaternion(qx, qy, qz, qw),
        "qw": qw, "qx": qx, "qy": qy, "qz": qz,
        "age_ms": round(float(age_ms), 2),
    }
    if camera is not None:
        camera = np.asarray(camera, dtype=np.float64)
        cw, cx, cy, cz = quat_from_matrix(camera[:3, :3])
        block = {
            "frame": str(child_frame),
            "child_frame": CAMERA_FRAME,
            "x": float(camera[0, 3]), "y": float(camera[1, 3]), "z": float(camera[2, 3]),
            "qw": float(cw), "qx": float(cx), "qy": float(cy), "qz": float(cz),
        }
        block.update(camera_info or {})
        pose["camera"] = block
    return pose
