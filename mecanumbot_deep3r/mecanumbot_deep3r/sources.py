"""
Feeding the RoboCam client from ROS subscriptions.

The client takes its data from *sources* — small objects with an iterator and an
``info()`` — because it was written to run standalone on the Orin with a serial
lidar and no ROS at all. These are the ROS-backed implementations of the three
the map loop needs: the robot's pose, its occupancy grid and its scan.

They are duck-typed rather than subclassed, for the same reason
``RosImageSource`` is: the client is a single file deployed by ``scp``, not a
package this one can import at module scope.  The contract is
``info()``/``<items>()``/``request_stop()``/``close()``, and the readings are
built by calling into the client module that was loaded by path.

**The pose is read from TF, not from a topic.** This is the one design decision
in here and it is not a preference. The server refuses to compare a pose against
a grid drawn in a different frame — a pose in ``odom`` is dead reckoning and a
pose in ``map`` is that plus SLAM's correction, and they differ by a metre after
a few minutes on carpet. So the pose must be in ``map``. But T1 runs under
slam_toolbox, which publishes **no** ``/amcl_pose``; the only map-frame pose
available during exploration is the ``map -> base_link`` transform. Reading TF
works unchanged under slam_toolbox in T1 and AMCL in T2, which is also what
``mecanumbot_autoslam`` does and for the same reason.

Sending ``/odom`` instead would not fail loudly. The server would reject every
pose as ``bad_odom``, the comparison would never run, ``mean_agreement`` would
stay ``null``, and T1 would simply never finish — with nothing in the robot's
logs pointing at the frame.
"""

from __future__ import annotations

import math
import threading

import numpy as np
from rclpy.duration import Duration

from . import camera_pose
from .geometry import yaw_from_quaternion


class _Queued:
    """
    One-slot handover from a ROS callback to the client's iterator thread.

    One slot, and the newest wins.  Every one of these carries a *current
    state* — where the robot is, what the map looks like — and an old one is
    not worth sending: the server compares against now, and a queue of stale
    poses would place clouds where the robot used to be.
    """

    def __init__(self):
        self._value = None
        self._event = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.dropped = 0

    def put(self, value):
        with self._lock:
            if self._value is not None:
                self.dropped += 1
            self._value = value
        self._event.set()

    def items(self, timeout=0.5):
        while not self._stop.is_set():
            if not self._event.wait(timeout):
                continue
            self._event.clear()
            with self._lock:
                value, self._value = self._value, None
            if value is not None:
                yield value

    def request_stop(self):
        self._stop.set()
        self._event.set()


class TfOdomSource:
    """
    Poses in the ``map`` frame, sampled from TF at a fixed rate.

    Sampled rather than event-driven because TF has no "new transform" event to
    subscribe to, and because the server wants a steady trickle rather than
    whatever rate the localiser happens to publish at.

    A lookup that fails is **skipped, not substituted**.  Early in a run, and for
    a while after a SLAM reset, ``map -> base_link`` does not exist; sending the
    last known pose then would tell the server the robot is standing still
    somewhere it has left, and every cloud compared against it would be placed
    wrongly with full confidence.  Silence is the honest answer, and the server
    already treats a missing pose as "nothing could be placed".
    """

    def __init__(self, node, client_module, buffer, *, target_frame="map",
                 source_frame="mecanumbot/base_link", rate_hz=10.0):
        self.node = node
        self._client = client_module
        self._buffer = buffer
        self.target_frame = str(target_frame)
        self.source_frame = str(source_frame)
        self.rate_hz = float(rate_hz)
        self._stop = threading.Event()
        self.lookups_failed = 0
        self.poses_read = 0
        self._warned = False

    def info(self):
        return {"source": "tf", "frame": self.target_frame,
                "child": self.source_frame, "rate_hz": self.rate_hz}

    def poses(self):
        period = 1.0 / max(1e-3, self.rate_hz)
        while not self._stop.is_set():
            if self._stop.wait(period):
                return
            reading = self._lookup()
            if reading is not None:
                yield reading

    def _lookup(self):
        from rclpy.time import Time
        try:
            tf = self._buffer.lookup_transform(
                self.target_frame, self.source_frame, Time(),
                timeout=Duration(seconds=0.0),
            )
        except Exception as exc:  # tf2 raises several unrelated types
            self.lookups_failed += 1
            if not self._warned:
                self._warned = True
                self.node.get_logger().warn(
                    f"no {self.target_frame} -> {self.source_frame} transform yet "
                    f"({exc}); the server cannot place the cloud until there is one"
                )
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        self.poses_read += 1
        if self._warned:
            self._warned = False
            self.node.get_logger().info(
                f"{self.target_frame} -> {self.source_frame} is available again")
        return self._client.OdomReading(
            x=t.x, y=t.y, z=t.z,
            yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
            quaternion=(q.x, q.y, q.z, q.w),
            frame=self.target_frame,
            child_frame=self.source_frame,
        )

    def request_stop(self):
        self._stop.set()

    def close(self):
        self.request_stop()


class NeckTracker:
    """
    The neck's last reported position, from ``opencr_state``.

    ``pos_n`` is the servo's **goal**: the firmware echoes the last command back
    and never reads the AX-12A's present position.  The head this describes is
    where it was told to be, which it reaches a servo's travel time later.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ticks = None
        self._stamp_ns = 0
        self.readings = 0

    def submit(self, msg):
        stamp_ns = (int(msg.header.stamp.sec) * 1_000_000_000
                    + int(msg.header.stamp.nanosec))
        with self._lock:
            self._ticks = int(msg.pos_n)
            self._stamp_ns = stamp_ns
            self.readings += 1

    def at(self, stamp_ns, stale_s):
        """
        Return ``(ticks, age_ms)`` for an image taken at ``stamp_ns``, or None.

        None when there has been no reading, or the newest is further than
        ``stale_s`` from the image in either direction: a board that stopped
        publishing leaves a plausible last value behind, and that is the one to
        refuse.
        """
        with self._lock:
            ticks, reading_ns = self._ticks, self._stamp_ns
        if ticks is None:
            return None
        age_ms = (int(stamp_ns) - reading_ns) / 1e6
        if abs(age_ms) > float(stale_s) * 1000.0:
            return None
        return ticks, age_ms


class FramePoser:
    """
    The pose to attach to one frame: where the base and the camera were.

    Called on the client's thread as each frame is handed over, with the
    image's own ROS stamp.  Returns the dict ``robocam.wire.frame(pose=...)``
    describes, or None -- sent as "no pose for this frame", which a server that
    has seen this robot's camera pose answers by not placing it.  The rule is
    ``TfOdomSource``'s: a pose that cannot be had is skipped, never substituted.

    **The base** is ``map -> base`` at the image's stamp: ``odom -> base``
    interpolated to that instant and ``map -> odom`` at its latest, which is
    ``lookup_transform_full`` with ``odom`` as the fixed frame.  Dead reckoning
    is what moves between frames and is published fast enough to interpolate;
    SLAM's correction moves slowly and is published late, so asking for it at
    the image's stamp would fail on nearly every frame.  The 10 Hz stream, by
    contrast, pairs a frame with whichever pose reached the server last.

    **The camera** is ``camera_pose.NeckCamera`` at the neck's last position, or
    nothing when ``camera`` is None -- not TF, for the reason that module gives.
    """

    def __init__(self, node, buffer, neck, camera, *, map_frame="map",
                 odom_frame="mecanumbot/odom", base_frame="mecanumbot/base_link",
                 lookup_timeout_s=0.05, neck_stale_s=0.5):
        self.node = node
        self._buffer = buffer
        self.neck = neck
        self.camera = camera
        self.map_frame = str(map_frame)
        self.odom_frame = str(odom_frame)
        self.base_frame = str(base_frame)
        self.lookup_timeout_s = float(lookup_timeout_s)
        self.neck_stale_s = float(neck_stale_s)
        self.poses = 0
        self.lookups_failed = 0
        self.neck_missing = 0
        self.neck_implausible = 0
        self._warned = set()

    def _warn_once(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            self.node.get_logger().warn(text)

    def __call__(self, stamp_ns):
        from rclpy.time import Time

        camera = None
        info = None
        age_ms = 0.0
        if self.camera is not None:
            reading = self.neck.at(stamp_ns, self.neck_stale_s)
            if reading is None:
                self.neck_missing += 1
                self._warn_once(
                    "neck",
                    f"no neck position within {self.neck_stale_s:.2f} s of a frame; "
                    "frames go without a pose, and so are not placed, until there is one")
                return None
            ticks, age_ms = reading
            camera = self.camera.extrinsic(ticks)
            if camera is None:
                self.neck_implausible += 1
                self._warn_once(
                    "implausible",
                    f"neck at {ticks} ticks is a pitch of "
                    f"{math.degrees(self.camera.pitch(ticks)):.0f} deg, which the head "
                    "cannot have; those frames go without a pose")
                return None
            info = {"source": "neck_model", "neck_ticks": int(ticks),
                    "pitch_deg": round(math.degrees(self.camera.pitch(ticks)), 2)}

        try:
            tf = self._buffer.lookup_transform_full(
                self.map_frame, Time(),
                self.base_frame, Time(nanoseconds=int(stamp_ns)),
                self.odom_frame, timeout=Duration(seconds=self.lookup_timeout_s),
            )
        except Exception as exc:  # tf2 raises several unrelated types
            self.lookups_failed += 1
            self._warn_once(
                "tf",
                f"no {self.map_frame} -> {self.base_frame} at a frame's stamp ({exc}); "
                "such frames go without a pose")
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        self.poses += 1
        return camera_pose.frame_pose_header(
            (t.x, t.y, t.z), (q.x, q.y, q.z, q.w),
            frame=self.map_frame, child_frame=self.base_frame,
            age_ms=age_ms, camera=camera, camera_info=info,
        )


class TopicMapSource:
    """
    The occupancy grid, from ``nav_msgs/OccupancyGrid``.

    ``map_id`` is derived from the grid's own geometry rather than invented,
    because it has to mean "the same map" across a reconnect: the client re-sends
    the last grid when a session comes back, and a fresh random id would make the
    server treat a continuing map as a new one and drop every patch keyed on the
    old one.  Origin, resolution and size change together exactly when
    slam_toolbox re-rasterises, which is the event that *should* change the id.
    """

    def __init__(self, node, client_module):
        self.node = node
        self._client = client_module
        self._queue = _Queued()
        self.maps_read = 0
        self.map_id = ""

    def info(self):
        return {"source": "ros2", "topic": "/map"}

    def submit(self, msg):
        info = msg.info
        cells = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        origin = (info.origin.position.x, info.origin.position.y,
                  yaw_from_quaternion(info.origin.orientation.x,
                                      info.origin.orientation.y,
                                      info.origin.orientation.z,
                                      info.origin.orientation.w))
        self.map_id = (f"{info.width}x{info.height}@{info.resolution:.4f}"
                       f"+{origin[0]:.3f},{origin[1]:.3f}")
        self.maps_read += 1
        self._queue.put(self._client.MapReading(
            cells=cells,
            resolution=float(info.resolution),
            origin=origin,
            frame=msg.header.frame_id or "map",
            map_id=self.map_id,
        ))

    def maps(self):
        yield from self._queue.items()

    def request_stop(self):
        self._queue.request_stop()

    def close(self):
        self.request_stop()


class TopicScanSource:
    """
    One lidar revolution, from ``sensor_msgs/LaserScan``.

    The scan is not needed for the comparison — that is the cloud against the
    grid — but it is what ``data.scale_check`` measures CUT3R's metric claim
    against, and that claim is what the whole approach rests on.  Cheap to send
    and the one number that catches a reconstruction that is confidently the
    wrong size.
    """

    def __init__(self, node, client_module):
        self.node = node
        self._client = client_module
        self._queue = _Queued()
        self.scans_read = 0
        self._info = {"source": "ros2", "topic": "/scan"}

    def info(self):
        return dict(self._info)

    def submit(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        # ScanReading is uint16 millimetres with 0 for "no return" -- the
        # LDS-02's own convention, so nothing converts on either side of the
        # wire.  inf and nan (LaserScan's "no return") map onto that 0, and so
        # does anything beyond 65 m, which would otherwise wrap round to a
        # plausible short range.
        valid = np.isfinite(ranges) & (ranges > 0.0)
        mm = np.where(valid, ranges * 1000.0, 0.0)
        mm = np.clip(mm, 0.0, 65535.0).astype(np.uint16)
        self.scans_read += 1
        self._info.update({"points": int(mm.size),
                           "angle_min": float(msg.angle_min),
                           "angle_max": float(msg.angle_max)})
        self._queue.put(self._client.ScanReading(
            ranges_mm=mm,
            angle_min=float(msg.angle_min),
            angle_increment=float(msg.angle_increment),
            range_min=float(msg.range_min),
            range_max=float(msg.range_max),
            scan_time=float(getattr(msg, "scan_time", 0.0)),
        ))

    def scans(self):
        yield from self._queue.items()

    def request_stop(self):
        self._queue.request_stop()

    def close(self):
        self.request_stop()
