"""
Stream camera frames to the off-board CUT3R server and republish the cloud.

The reconstruction runs on the cluster, not here: a monocular pointmap model
does not fit on the Orin alongside the rest of the stack.  This node is the
robot's end of that loop -- it takes the frames the camera is already
publishing, hands them to the RoboCam client, and turns what comes back into a
``PointCloud2``.

Why it subscribes rather than opening the camera
------------------------------------------------
``mecanumbot_camera_stream`` already owns the camera on the Orin.  A second
process opening ``/dev/video0`` would either fail or take the device away from
it, so this node consumes the compressed topic that node publishes.  That also
means the JPEG is already encoded: the frames are forwarded to the server
untouched, with no decode-re-encode round trip on the robot's CPU.

Timestamps
----------
The cloud has to be stamped with when the picture was *taken*, not when the
reply arrived -- a round trip is a couple of hundred milliseconds, and on a
moving robot a cloud stamped on arrival is placed where the robot is now
rather than where it was looking, which smears the map by exactly the distance
travelled during the round trip.

The client does not carry per-frame metadata, so the correlation is done on
its clock: every reply echoes ``t_send_ns``, a ``time.monotonic_ns()`` reading
from *this machine*, and this node records the same clock when it hands each
frame over.  Matching a reply to the most recent hand-over at or before its
``t_send_ns`` recovers the right image, and does so without assuming that
every frame handed over was sent -- the client drops frames at the source
under load, which would break any scheme based on counting.
"""

import collections
import importlib.util
import math
import os
import queue
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CompressedImage, LaserScan, PointCloud2, PointField
from std_msgs.msg import Bool, Header, String
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

from mecanumbot_msgs.msg import MapCloudAgreement, OpenCRState

from . import bridge
from . import camera_pose
from . import cloud as cloud_codec
from . import sources as ros_sources
from .geometry import quat_from_matrix


class RosImageSource:
    """
    Feed a RoboCam ``FrameSource`` from a ROS subscription.

    Duck-typed rather than subclassed: the client is a standalone file deployed
    from ``RoboCamStreamProcessing/link``, not a package this one can import at
    module scope.  The contract is small -- ``width``, ``height``, ``fps``,
    ``frames()`` and ``close()``.

    ``width``/``height`` are advertised metadata only; the server reports what
    it actually decoded rather than what the client claimed, so they do not
    have to be right for the pipeline to be correct.
    """

    def __init__(self, width, height, fps, queue_depth=2):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        # One slot more than the client's in-flight window is plenty: a frame
        # waiting here is a frame going stale, and the robot would rather send
        # the newest one than a queued old one.
        self._q = queue.Queue(maxsize=max(1, queue_depth))
        self._stop = threading.Event()
        #: (monotonic_ns at hand-over, ROS stamp ns) for reply correlation.
        self.handovers = collections.deque(maxlen=256)
        self.dropped = 0
        #: Called with each frame's stamp at hand-over; its answer rides on the
        #: frame.  None sends frames exactly as before protocol 2's frame pose.
        self.poser = None

    def submit(self, jpeg, stamp_ns):
        """Offer a frame; drop the oldest rather than block the ROS executor."""
        try:
            self._q.put_nowait((jpeg, stamp_ns))
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait((jpeg, stamp_ns))
            except queue.Empty:  # pragma: no cover - lost the race, harmless
                pass
            self.dropped += 1

    def frames(self):
        while not self._stop.is_set():
            try:
                jpeg, stamp_ns = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            # The pose first: its TF lookup may wait a few tens of milliseconds
            # for odometry to catch up with the image, and the hand-over time
            # should be the one closest to the send.
            pose = self.poser(stamp_ns) if self.poser is not None else None
            self.handovers.append((time.monotonic_ns(), stamp_ns))
            # (None, bytes) means "already encoded" -- see the client's CSI
            # source, which does the same with its hardware-encoded frames.
            if self.poser is None:
                yield None, jpeg
            else:
                yield None, jpeg, pose

    def close(self):
        self._stop.set()


def _load_client(path):
    """
    Import the standalone client file from wherever it was deployed.

    It is one file with no package around it, on purpose -- that is what makes
    it deployable to the robot with a single ``scp`` -- so it is loaded by
    path rather than imported by name.
    """
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        path = os.path.join(path, "robocam_client.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"robocam_client.py not found at {path}. Deploy it from "
            "RoboCamStreamProcessing/link and set the client_path parameter."
        )
    spec = importlib.util.spec_from_file_location("robocam_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bridge.check_client_api(module, path)
    return module


class Deep3RClientNode(Node):
    """Frames out to the server, ``PointCloud2`` back in."""

    def __init__(self):
        super().__init__("mecanumbot_deep3r_client_node")
        self.declare_parameters(
            namespace="",
            parameters=[
                # 127.0.0.1 by default, not the server's own address: the robot
                # cannot route to it. mecanumbot-deep3r-tunnel.service forwards
                # the port here. See RoboCamStreamProcessing/link/README.md.
                ("server", "tcp://127.0.0.1:5555"),
                ("client_path", "~/server/RoboCamStreamProcessing/link/robocam_client.py"),
                ("camera_topic", "camera/image_raw/compressed"),
                ("cloud_topic", "deep3r/points"),
                ("pose_topic", "deep3r/pose"),
                ("publish_pose", True),
                # The cloud is in CUT3R's own world frame, anchored on the first
                # frame of each map_id. It is NOT yet connected to the robot's
                # TF tree -- see this package's README.
                ("frame_id", "deep3r_world"),
                ("advertised_width", 1280),
                ("advertised_height", 720),
                ("advertised_fps", 15.0),
                ("max_inflight", 2),
                ("queue_depth", 2),
                ("log_every", 30),

                # --- the map loop (protocol 2) ---------------------------
                # Everything below turns this node from a frame pump into the
                # robot's half of the map loop.  See docs/INTEGRATION.md in
                # RoboCamStreamProcessing for what each side expects.
                ("enable_map_loop", True),
                ("map_topic", "/map"),
                ("scan_topic", "/scan"),
                # Poses come from TF, not from a topic: T1 runs under
                # slam_toolbox, which publishes no /amcl_pose, and the server
                # refuses to compare a pose against a grid in another frame.
                ("map_frame", "map"),
                ("base_frame", "mecanumbot/base_link"),
                ("pose_rate_hz", 10.0),
                # The pose attached to each frame: the base at the image's own
                # stamp, and the camera from the neck.  See camera_pose.py for
                # why the camera is a model and not a TF lookup.
                ("send_frame_pose", True),
                ("odom_frame", "mecanumbot/odom"),
                ("pose_lookup_timeout_s", 0.05),
                ("send_camera_pose", True),
                ("neck_topic", "opencr_state"),
                ("neck_stale_s", 0.5),
                ("camera.pivot_x", 0.1063),
                ("camera.pivot_z", 0.1679),
                ("camera.lever_x", 0.022),
                ("camera.lever_z", 0.038),
                ("camera.level_ticks", 600.0),
                ("camera.rad_per_tick", 0.005061),
                ("camera.pitch_at_level_deg", 0.0),
                ("map_every_s", 5.0),
                # What the robot publishes from what comes back.
                ("agreement_topic", "/mecanumbot/deep3r/map_agreement"),
                ("seek_target_topic", "/mecanumbot/seek/target"),
                ("seek_detections_topic", "/mecanumbot/seek/detections"),
                # T1 -> T2 handover.  The explorer latches this when its exit
                # criteria are met; the server never changes phase on its own,
                # so without this the two spend the rest of the run in
                # different phases.
                ("finished_topic", "/mecanumbot/exploration/finished"),
                ("request_topic", "/mecanumbot/seek/request"),
                # A pose_hint is advisory and nothing here applies one:
                # slam_toolbox owns the robot's pose and has a graph the server
                # cannot see.  These only gate whether it is worth logging.
                ("hint_min_confidence", 0.5),
                ("hint_min_inliers", 40),
                # Which run this is. Empty means "mint a fresh one", which is
                # what you want: the server wipes its reconstruction, keyframes
                # and remembered target when this changes, so relaunching the
                # robot gives it a clean slate rather than folding a new pass
                # into the last room's state. A reconnect after a dropped
                # tunnel keeps the same value and does not wipe. Set it
                # explicitly only to *resume* a run across a node restart.
                ("run_id", ""),
            ],
        )
        gp = self.get_parameter
        self.frame_id = str(gp("frame_id").value)
        self.publish_pose = bool(gp("publish_pose").value)
        self.log_every = int(gp("log_every").value)

        self.source = RosImageSource(
            gp("advertised_width").value,
            gp("advertised_height").value,
            gp("advertised_fps").value,
            queue_depth=int(gp("queue_depth").value),
        )
        # Results are handed to the executor rather than published from the
        # client's thread, so every publish happens on one thread and the
        # node's clock is read from the same place it always is.
        self._results = queue.Queue(maxsize=8)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CompressedImage, str(gp("camera_topic").value),
            self._on_image, sensor_qos,
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, str(gp("cloud_topic").value), 1)
        self.pose_pub = (
            self.create_publisher(PoseStamped, str(gp("pose_topic").value), 10)
            if self.publish_pose else None
        )
        self.create_timer(0.02, self._drain_results)

        client_module = _load_client(str(gp("client_path").value))
        self._client_module = client_module

        self.map_loop = bool(gp("enable_map_loop").value)
        self._announcements = queue.Queue(maxsize=32)
        if self.map_loop:
            self._setup_map_loop(gp, client_module)

        self.client = client_module.RoboCamClient(
            server=str(gp("server").value),
            client_id=f"mecanumbot-{os.uname().nodename}",
            max_inflight=int(gp("max_inflight").value),
            run_id=(str(gp("run_id").value) or None),
            on_result=self._on_result,
            map_every_s=float(gp("map_every_s").value),
            # The three the server sends unasked.  Handed straight to a queue:
            # they arrive on the client's thread and every ROS publish in this
            # node happens on the executor's.
            on_map_update=(self._queue_announcement if self.map_loop else None),
            on_pose_hint=(self._queue_announcement if self.map_loop else None),
            on_found=(self._queue_announcement if self.map_loop else None),
            on_agreement=(self._queue_announcement if self.map_loop else None),
        )
        self._frames_in = 0
        self._clouds_out = 0
        self._last_cloud_map_id = None
        self._thread = threading.Thread(target=self._run_client, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"deep3r client -> {gp('server').value}, frames from "
            f"{gp('camera_topic').value}, cloud on {gp('cloud_topic').value} "
            f"in frame '{self.frame_id}'"
            + (f"; map loop on ({gp('map_topic').value}, {gp('scan_topic').value}, "
               f"pose from {gp('map_frame').value} -> {gp('base_frame').value})"
               if self.map_loop else "; map loop OFF")
        )
        # Logged because it is the join between this robot's logs and the
        # server's: the server reports the same string when it wipes, so a run
        # that started against stale state is visible from either end.
        self.get_logger().info(
            f"run {self.client.run_id} -- the server clears its reconstruction, "
            "keyframes and remembered target for a run it has not seen")

    def _setup_map_loop(self, gp, client_module):
        """
        Wire the uplink sources and the downlink publishers.

        Split out of ``__init__`` because it is a whole subsystem and because
        ``enable_map_loop:=false`` has to leave this node exactly as it was
        before protocol 2 -- a frame pump -- for benchmarking the transport
        against a server with compare disabled.
        """
        from tf2_ros import Buffer, TransformListener

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.odom_source = ros_sources.TfOdomSource(
            self, client_module, self._tf_buffer,
            target_frame=str(gp("map_frame").value),
            source_frame=str(gp("base_frame").value),
            rate_hz=float(gp("pose_rate_hz").value),
        )
        self.map_source = ros_sources.TopicMapSource(self, client_module)
        self.scan_source = ros_sources.TopicScanSource(self, client_module)

        # The map is latched by slam_toolbox and by map_server, so a subscriber
        # that misses the last publish would otherwise wait for the next one --
        # which, on a saved map in T2, never comes.
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._setup_frame_pose(gp, client_module, sensor_qos)
        self.create_subscription(
            OccupancyGrid, str(gp("map_topic").value),
            lambda msg: self.map_source.submit(msg), latched)
        self.create_subscription(
            LaserScan, str(gp("scan_topic").value),
            lambda msg: self.scan_source.submit(msg), sensor_qos)
        self.create_subscription(
            Bool, str(gp("finished_topic").value), self._on_finished, latched)
        self.create_subscription(
            String, str(gp("request_topic").value), self._on_request, 10)

        self.agreement_pub = self.create_publisher(
            MapCloudAgreement, str(gp("agreement_topic").value), 10)
        self.seek_target_pub = self.create_publisher(
            Detection3DArray, str(gp("seek_target_topic").value), 10)
        self.seek_detections_pub = self.create_publisher(
            Detection3DArray, str(gp("seek_detections_topic").value), 10)

        self.hint_min_confidence = float(gp("hint_min_confidence").value)
        self.hint_min_inliers = int(gp("hint_min_inliers").value)
        self._phase = "t1"
        self._target = ""
        self._agreements_out = 0
        self._founds_out = 0
        self.create_timer(0.05, self._drain_announcements)

    def _setup_frame_pose(self, gp, client_module, sensor_qos):
        """
        Attach the robot's pose, and its camera's, to every frame.

        Part of the map loop because it needs the TF buffer and is meaningless
        without a map to place clouds in.  A client file too old to carry it
        costs the per-frame pose and nothing else, so it is a warning and not
        the refusal ``check_client_api`` gives a client too old for the loop.
        """
        self.frame_poser = None
        if not bool(gp("send_frame_pose").value):
            return
        if not bridge.client_takes_frame_pose(client_module):
            self.get_logger().warn(
                "the deployed robocam_client.py cannot attach a pose to a frame, so "
                "clouds are placed with the 10 Hz pose and the server's fixed camera "
                "mount whatever the neck is doing. Redeploy it from "
                "RoboCamStreamProcessing/link.")
            return

        camera = None
        self.neck = None
        if bool(gp("send_camera_pose").value):
            camera = camera_pose.NeckCamera(
                pivot_x=float(gp("camera.pivot_x").value),
                pivot_z=float(gp("camera.pivot_z").value),
                lever_x=float(gp("camera.lever_x").value),
                lever_z=float(gp("camera.lever_z").value),
                level_ticks=float(gp("camera.level_ticks").value),
                rad_per_tick=float(gp("camera.rad_per_tick").value),
                pitch_at_level=math.radians(float(gp("camera.pitch_at_level_deg").value)),
            )
            self.neck = ros_sources.NeckTracker()
            self.create_subscription(
                OpenCRState, str(gp("neck_topic").value), self.neck.submit, sensor_qos)
            # Logged, as perception logs its camera height: a run should say what
            # it assumed about where the camera was.
            self.get_logger().info(camera.describe())

        self.frame_poser = ros_sources.FramePoser(
            self, self._tf_buffer, self.neck, camera,
            map_frame=str(gp("map_frame").value),
            odom_frame=str(gp("odom_frame").value),
            base_frame=str(gp("base_frame").value),
            lookup_timeout_s=float(gp("pose_lookup_timeout_s").value),
            neck_stale_s=float(gp("neck_stale_s").value),
        )
        self.source.poser = self.frame_poser

    # -- plumbing ------------------------------------------------------------

    def _run_client(self):
        kwargs = {}
        if self.map_loop:
            # The pose, the grid and the scan go up the same socket as the
            # frames.  Without them the server has nothing to place the cloud
            # against and every comparison is skipped -- silently, because "no
            # pose" and "compared and found nothing" look identical from here.
            kwargs = {
                "odom_source": self.odom_source,
                "map_source": self.map_source,
                "scan_source": self.scan_source,
            }
        try:
            self.client.run(self.source, status_every=0, **kwargs)
        except Exception as exc:  # pragma: no cover - reported, not raised
            self.get_logger().error(f"RoboCam client stopped: {exc}")

    def _on_image(self, msg):
        self._frames_in += 1
        stamp_ns = (int(msg.header.stamp.sec) * 1_000_000_000
                    + int(msg.header.stamp.nanosec))
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        self.source.submit(bytes(msg.data), stamp_ns)

    def _on_result(self, result):
        """Take one reply on the client's thread; do no ROS work here."""
        try:
            self._results.put_nowait(result)
        except queue.Full:
            pass  # a stale cloud is worth less than the next one

    # -- publishing ----------------------------------------------------------

    def _drain_results(self):
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return
            try:
                self._publish(result)
            except cloud_codec.CloudFormatError as exc:
                self.get_logger().warn(f"undecodable cloud: {exc}")

    def _publish(self, result):
        data = result.get("data") or {}
        if not cloud_codec.has_cloud(data):
            return
        points, colors = cloud_codec.decode(data["cloud"])
        stamp = self._stamp_for(result)

        # CUT3R's reconstruction session, NOT the robot's SLAM map -- the two
        # are different identities that invalidate different things, and they
        # were briefly the same field name.  A change here restarts the cloud's
        # world frame; a change in the robot's map_id (see bridge.is_stale)
        # invalidates map-frame coordinates instead.
        cloud_map_id = data.get("map_id")
        if cloud_map_id != self._last_cloud_map_id:
            if self._last_cloud_map_id is not None:
                self.get_logger().warn(
                    f"cloud_map_id {self._last_cloud_map_id} -> {cloud_map_id} "
                    f"({data.get('reset_reason', 'reset')}): the reconstruction's "
                    "world frame restarted, discard whatever was accumulated "
                    "against the previous one"
                )
            self._last_cloud_map_id = cloud_map_id

        if points.shape[0]:
            self.cloud_pub.publish(self._to_msg(points, colors, stamp))
            self._clouds_out += 1

        pose = data.get("pose_c2w")
        if self.pose_pub is not None and pose:
            self.pose_pub.publish(self._to_pose(np.asarray(pose), stamp))

        if self.log_every and self._clouds_out % self.log_every == 1:
            check = data.get("scale_check") or {}
            poser = getattr(self, "frame_poser", None)
            self.get_logger().info(
                f"cloud map {cloud_map_id} frame {data.get('frames_in_state')}: "
                f"{points.shape[0]} pts, infer {data.get('infer_ms')} ms, "
                f"in {self._frames_in} dropped {self.source.dropped}"
                + (f", lidar/cloud ratio {check['ratio']}" if check.get("ratio")
                   else "")
                + (f", pose {result.get('pose_source') or 'none'}"
                   f" (frame poses {poser.poses}, tf misses {poser.lookups_failed},"
                   f" neck missing {poser.neck_missing})" if poser is not None else "")
            )

    # -- the map loop --------------------------------------------------------

    def _queue_announcement(self, header):
        """Take one announcement on the client's thread; do no ROS work here."""
        try:
            self._announcements.put_nowait(header)
        except queue.Full:
            # Verdicts are periodic and the next one supersedes this one; a
            # `found` is not, so it is worth saying that one was lost.
            if header.get("type") == "found":
                self.get_logger().warn("dropped a `found` announcement: queue full")

    def _drain_announcements(self):
        while True:
            try:
                header = self._announcements.get_nowait()
            except queue.Empty:
                return
            try:
                self._dispatch(header)
            except bridge.BridgeError as exc:
                # Malformed rather than merely unwelcome.  Logged and dropped:
                # this is the server and the robot disagreeing about the
                # contract, and guessing at the intent is how a wrong region
                # ends up in the costmap looking like a right one.
                self.get_logger().error(
                    f"unusable {header.get('type', '?')} announcement: {exc}")

    def _dispatch(self, header):
        kind = header.get("type")
        if kind == "agreement":
            self._on_agreement(header)
        elif kind == "found":
            self._on_found(header)
        elif kind == "pose_hint":
            self._on_pose_hint(header)
        elif kind == "map_update":
            # Deliberately not applied here.  The patch belongs in the costmap,
            # and `mecanumbot_map_agreement` is what owns that -- it rebuilds a
            # keepout mask from the verdict's regions, with a height decision
            # this node has no business making.  Merging the cells here as well
            # would put the same obstacles into the map twice, by two routes
            # with different lifetimes.
            self.get_logger().debug(
                f"map_update: {header.get('cells_changed', 0)} cells "
                f"(applied via the agreement verdict, not merged here)")
        else:
            self.get_logger().warn(f"unknown announcement type {kind!r}")

    def _on_agreement(self, header):
        """
        Republish the server's comparison verdict as MapCloudAgreement.

        This is the message the robot's exploration runs on: the frontier
        explorer's `CLOUD` exit criterion and `mecanumbot_map_agreement`'s
        keepout mask are both built from it.  Until this node published it,
        neither could ever be satisfied and T1 could not finish.
        """
        if bridge.is_stale(header, self._current_map_id()):
            self.get_logger().info(
                f"dropping a verdict for map_id {header.get('map_id')!r}; "
                f"the robot is on {self._current_map_id()!r}")
            return

        regions = bridge.agreement_regions(header)
        msg = MapCloudAgreement()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(header.get("frame", "map"))
        # Both identities, under names that say which is which.  `map_id` is
        # the robot's SLAM map -- a change there means these coordinates name a
        # place in a map that no longer exists.  `cloud_map_id` is CUT3R's
        # session -- a change there voids what was derived from the cloud's
        # geometry and nothing else.  They were one field once, and the two
        # consumers of this message need different ones.
        msg.map_id = str(header.get("map_id", "") or "")
        cloud_id = bridge.cloud_map_id(header)
        msg.cloud_map_id = "" if cloud_id is None else str(cloud_id)
        msg.cloud_points = int(header.get("cloud_points", 0))
        msg.grid_coverage = float(header.get("grid_coverage", 0.0))
        msg.cloud_coverage = float(header.get("cloud_coverage", 0.0))
        msg.agreement = float(header.get("agreement", 0.0))
        msg.compared_cells = int(header.get("compared_cells", 0))
        msg.conflicting_cells = int(header.get("conflicting_cells", 0))

        for region in regions:
            point = Point()
            point.x = region["x"]
            point.y = region["y"]
            # z carries the structure height, which is what the handler's whole
            # decision turns on.  A region with no measurable height is sent as
            # NaN so the handler's "unknown means blocking" branch fires --
            # 0.0 would read as "on the floor" and be ignored as a threshold
            # strip, which is the one reading that drives the robot into it.
            point.z = (float("nan") if region["height"] is None
                       else float(region["height"]))
            msg.uncertain_points.append(point)
            msg.uncertain_scores.append(float(region["score"]))
            msg.uncertain_kinds.append(str(region["kind"]))
            msg.uncertain_heights.append(
                float("nan") if region["height"] is None else float(region["height"]))
            msg.uncertain_radii.append(float(region["radius"]))

        self.agreement_pub.publish(msg)
        self._agreements_out += 1
        if self.log_every and self._agreements_out % self.log_every == 1:
            self.get_logger().info(
                f"agreement {msg.agreement:.2f} | grid coverage "
                f"{msg.grid_coverage:.2f} | {len(regions)} regions")

    def _on_found(self, header):
        """
        Route the decision stage's answer onto one of the seek tree's inputs.

        `live` is perception and `memory` is a memory, and the tree's two
        parallel branches are exactly that distinction -- so they go to two
        topics.  `absent` goes to neither: an empty array would be read as "no
        detection this frame", which is also what a blank wall produces, and
        the tree would lose the difference between not seeing the object and
        having looked for it.
        """
        if bridge.is_stale(header, self._current_map_id()):
            self.get_logger().warn(
                f"dropping a `found` for map_id {header.get('map_id')!r}; "
                f"the robot is on {self._current_map_id()!r} and those "
                "coordinates name a place in a map that no longer exists")
            return

        topic = bridge.found_topic(header)
        hyp = bridge.found_hypothesis(header)
        if topic is None:
            self.get_logger().info(
                f"the server looked for {hyp['class_id']!r} and it is not here")
            return

        msg = Detection3DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(header.get("frame", "map"))

        detection = Detection3D()
        detection.header = msg.header
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = hyp["class_id"]
        result.hypothesis.score = hyp["score"]
        result.pose.pose.position.x = hyp["x"]
        result.pose.pose.position.y = hyp["y"]
        # The height, and the reason this system has a reconstruction at all:
        # it is what CheckObjectGraspable reads to tell a mug on a table from a
        # mug on the floor behind a chair.
        result.pose.pose.position.z = hyp["z"]
        result.pose.pose.orientation.w = 1.0
        detection.results.append(result)
        detection.bbox.center.position.x = hyp["x"]
        detection.bbox.center.position.y = hyp["y"]
        detection.bbox.center.position.z = hyp["z"]
        detection.bbox.center.orientation.w = 1.0
        msg.detections.append(detection)

        if topic == "target":
            self.seek_target_pub.publish(msg)
        else:
            self.seek_detections_pub.publish(msg)
        self._founds_out += 1
        self.get_logger().info(
            f"{hyp['basis']} sighting of {hyp['class_id']!r} at "
            f"({hyp['x']:.2f}, {hyp['y']:.2f}, {hyp['z']:.2f}) "
            f"confidence {hyp['score']:.2f} -> seek/{topic}"
            + (f"; server says {hyp['reach_verdict']}" if hyp["reach_verdict"] else ""))

    def _on_pose_hint(self, header):
        """
        Log a correction the server is offering SLAM.  Nothing applies it.

        The hint is advisory in the protocol and stays advisory here:
        slam_toolbox owns `map -> odom`, has a pose graph and loop closures the
        server cannot see, and a node that quietly re-broadcast the transform
        would be a second thing estimating it.  It is worth surfacing because a
        run where the server keeps offering the same correction is a run with a
        systematic placement error -- usually a wrong camera height.
        """
        if not bridge.pose_hint_is_usable(header, self.hint_min_confidence,
                                          self.hint_min_inliers):
            return
        self.get_logger().info(
            f"the server would move the cloud by ({header.get('dx', 0.0):+.3f}, "
            f"{header.get('dy', 0.0):+.3f}) m to agree with the map "
            f"({header.get('inliers', 0)} inliers) -- advisory, not applied")

    def _on_finished(self, msg):
        """
        T1 -> T2, on the explorer's latch.

        The explorer decides when T1 is over and latches it; this node is what
        tells the server.  The server never changes phase on its own -- that is
        a decision about the mission -- so without this the robot spends T2
        seeking while the server is still exploring, and the decision stage
        that produces `found` never runs.
        """
        if not msg.data or self._phase == "t2":
            return
        self._phase = "t2"
        self.get_logger().info(
            "exploration finished -> asking the server for phase t2"
            + (f", target {self._target!r}" if self._target else
               " with no target named yet"))
        try:
            self.client.set_phase(
                "t2", reason="exploration/finished latched",
                mission=({"target": self._target} if self._target else None))
        except Exception as exc:  # pragma: no cover - link down, retried below
            self.get_logger().error(f"could not set phase: {exc}")
            self._phase = "t1"

    def _on_request(self, msg):
        """
        Forward what the robot has been asked to find to the server.

        Free text, not a class label: it is the query an open-vocabulary
        detector is given, and "the red mug on the desk" finds what "mug" does
        not.  The robot's target wins over one named at the server's launch,
        because the robot is where the mission is being run from.
        """
        target = msg.data.strip()
        if not target or target == self._target:
            return
        self._target = target
        self.get_logger().info(f"target is now {target!r}")
        if self._phase == "t2":
            try:
                self.client.set_phase("t2", reason="target changed",
                                      mission={"target": target})
            except Exception as exc:  # pragma: no cover
                self.get_logger().error(f"could not update the target: {exc}")

    def _current_map_id(self):
        """Return the robot's SLAM map identity, as the grid source derives it."""
        return getattr(getattr(self, "map_source", None), "map_id", "")

    def _stamp_for(self, result):
        """
        Recover the ROS stamp of the image this reply belongs to.

        Falls back to now when the reply carries no ``t_send_ns`` or predates
        every hand-over this node remembers -- which happens for the first
        replies after a reconnect, and is better than refusing to publish.
        """
        sent = result.get("t_send_ns")
        if sent:
            best = None
            for handed, stamp_ns in self.source.handovers:
                if handed <= sent and (best is None or handed > best[0]):
                    best = (handed, stamp_ns)
            if best is not None:
                return rclpy.time.Time(nanoseconds=best[1]).to_msg()
        return self.get_clock().now().to_msg()

    def _to_msg(self, points, colors, stamp):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        packed = cloud_codec.to_xyzrgb(points, colors)
        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = packed.tobytes()
        return msg

    def _to_pose(self, c2w, stamp):
        msg = PoseStamped()
        msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        msg.pose.position.x = float(c2w[0, 3])
        msg.pose.position.y = float(c2w[1, 3])
        msg.pose.position.z = float(c2w[2, 3])
        qw, qx, qy, qz = quat_from_matrix(c2w[:3, :3])
        msg.pose.orientation.w = qw
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        return msg

    def destroy_node(self):
        self.source.close()
        # The three uplink sources each hold an iterator the client's thread is
        # blocked in; without this the node hangs on shutdown for as long as
        # their poll timeouts.
        for name in ("odom_source", "map_source", "scan_source"):
            source = getattr(self, name, None)
            if source is not None:
                source.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Deep3RClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
