# mecanumbot_deep3r

Off-board 3D reconstruction. The robot's camera frames go to the cluster, where
CUT3R turns them into a metric point cloud, and the cloud comes back and is
republished as `sensor_msgs/PointCloud2`.

This is the robot half of the "Deep3R" box in the system diagram. The model
half is the `deep3r` processor in `RoboCamStreamProcessing`; nothing in this
package imports torch, and the node runs on the Orin with numpy and pyzmq.

```text
  camera_stream ──► /camera/image_raw/compressed ──► this node
                                                       │  JPEG, forwarded as-is
                                                       ▼
                                              robocam_client (ZeroMQ)
                                                       │  tcp://127.0.0.1:5555
                                                       ▼  (forward SSH tunnel)
                                              nipg36: deep3r / CUT3R
                                                       │
  /mecanumbot/deep3r/points ◄── PointCloud2 ◄──────────┘
  /mecanumbot/deep3r/pose   ◄── PoseStamped
```

## Node: mecanumbot_deep3r_client_node

### Subscribers

| Topic | Data type | Processing |
| --- | --- | --- |
| `camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Forwarded to the server **as the JPEG it already is** — no decode/re-encode on the robot. |

### Publishers

| Topic | Data type | Function |
| --- | --- | --- |
| `deep3r/points` | `sensor_msgs/msg/PointCloud2` | The returned cloud: `xyz` + packed `rgb`, metres. |
| `deep3r/pose` | `geometry_msgs/msg/PoseStamped` | CUT3R's camera pose for that frame, same frame as the cloud. Disable with `publish_pose`. |

### Parameters

| Parameter | Default | Function |
| --- | --- | --- |
| `server` | `tcp://127.0.0.1:5555` | Local end of the forward tunnel — **not** the server's own address. |
| `client_path` | `~/robocam_client.py` | The deployed standalone client; a directory is also accepted. |
| `camera_topic` | `camera/image_raw/compressed` | Where frames come from. |
| `cloud_topic` | `deep3r/points` | Point cloud output. |
| `pose_topic` | `deep3r/pose` | Camera pose output. |
| `publish_pose` | `true` | Publish the pose alongside the cloud. |
| `frame_id` | `deep3r_world` | Frame the cloud and pose are stamped in. See the caveat below. |
| `advertised_width` / `_height` / `_fps` | `1280` / `720` / `15.0` | Metadata the client announces. The server reports what it actually decoded, so these need not be exact. |
| `max_inflight` | `2` | Frames awaiting a reply. |
| `queue_depth` | `2` | Frames buffered between the subscription and the client. |
| `log_every` | `30` | Log one line every N clouds; 0 disables. |
| `send_frame_pose` | `true` | Attach the robot's map pose at the image's stamp to every frame. See below. |
| `odom_frame` | `mecanumbot/odom` | Fixed frame for that lookup: `odom -> base` at the stamp, `map -> odom` at its latest. |
| `pose_lookup_timeout_s` | `0.05` | How long a frame waits for odometry to reach its stamp. |
| `send_camera_pose` | `true` | Attach the camera's pose from the neck model as well. |
| `neck_topic` / `neck_stale_s` | `opencr_state` / `0.5` | Where the neck position comes from, and how far from a frame it may be. |
| `camera.pivot_x` / `_z`, `camera.lever_x` / `_z` | `0.1063` / `0.1679`, `0.022` / `0.038` | The neck pivot and the pivot-to-lens lever, in `base_link`. |
| `camera.level_ticks` | `600.0` | Servo ticks of the trees' `neck_level_pos`. |
| `camera.rad_per_tick` | `0.005061` | Tilt per servo tick. |
| `camera.pitch_at_level_deg` | `0.0` | **Unmeasured.** Lens tilt at `level_ticks`, positive up. |

A few more subscriptions come with the map loop: `/map`, `/scan`, the explorer's
`finished` latch, the seek request, and `opencr_state` for the neck.

## Where the camera was, per frame

The server places each cloud with `T_map_base · T_base_camera`. It used to take
the first from the 10 Hz pose stream — whichever pose reached it last — and the
second from one fixed mount in its own config. On this robot both are wrong
while anything moves: the pose can be a tenth of a second away from the image,
and the camera sits on a neck that the fetch tree sweeps continuously.

So every frame now carries its own pose (`wire.frame(pose=...)` in
RoboCamStreamProcessing):

- **The base** is `map -> base_link` **at the image's stamp**, looked up with
  `odom` as the fixed frame: dead reckoning interpolated to that instant, SLAM's
  slow correction at its latest.
- **The camera** comes from a neck model in `camera_pose.py`: a pivot on the
  base, a lever to the lens that turns with the head, and a tilt linear in the
  neck's servo ticks. At level it puts the lens at (0.128, 0.206) m, matching
  the 0.13 / 0.21 that `mecanumbot_sensorprocess_smart` uses.

**It is not a TF lookup, and that is not an oversight.** The URDF's `head_link`
and `camera_link` are rotated 90° each for the meshes; composed,
`camera_rgb_optical_frame` looks along the neck's own axis — to the robot's
right — at every neck angle, so in TF the neck spins the image instead of
tilting it. Perception works around the same problem with measured parameters,
and this does the same. Fixing the URDF is the better long-term answer, but it
needs the neck's zero measured on the robot (`head_joint` in `joint_states` is
uncalibrated, and the simulator converts ticks with the opposite sign).

Two assumptions to know about:

- **`pitch_at_level_deg` is unmeasured.** Nothing establishes that the trees'
  "neutral driving gaze" (`neck_level_pos` 6.0) is optically level. A cloud
  that is consistently tilted, or a `pose_hint` that keeps offering the same
  correction, points here.
- **The neck position is the goal, not the head.** The firmware echoes the last
  command back as `opencr_state.pos_n` and never reads the AX-12A's present
  position, so during a sweep the model is ahead of the head by the servo's
  travel time.

A frame whose pose cannot be had — no TF at its stamp, no neck reading within
`neck_stale_s` — goes **without** one, rather than with the last good one. The
server, having seen a camera pose from this robot, does not place it. The log
line every `log_every` clouds says how many frames had a pose and why the rest
did not, and each result's `pose_source` says `frame` or `stream`; the server's
comparison stats say `camera_from: robot` or `config`.

This needs a `robocam_client.py` that can carry the pose. An older one still
runs the whole loop; the node says so once at startup and falls back to the
stream pose and the server's mount.

## Why it subscribes instead of opening the camera

`mecanumbot_camera_stream` already owns the camera device on the Orin. A second
process opening `/dev/video0` would either fail or take it away. Consuming the
compressed topic also means the JPEG is already encoded, so frames are
forwarded untouched rather than decoded and re-encoded on the robot's CPU.

## Timestamps

The cloud is stamped with when the **picture was taken**, not when the reply
arrived. A round trip is a couple of hundred milliseconds; a cloud stamped on
arrival is placed where the robot is *now* rather than where it was looking,
smearing the map by exactly the distance travelled in between.

The client carries no per-frame metadata, so the correlation runs on its clock.
Every reply echoes `t_send_ns`, a `time.monotonic_ns()` reading taken on this
same machine, and the node records that clock when it hands each frame over.
Matching a reply to the most recent hand-over at or before its `t_send_ns`
recovers the right image — and does so without assuming every frame handed over
was sent, which matters because the client drops frames at the source under
load. Anything based on counting frames would drift silently the first time it
did.

## The frame is not connected to TF yet

`frame_id` defaults to `deep3r_world`, and **nothing publishes a transform for
it.** The cloud arrives in CUT3R's own world frame, anchored on the first frame
of each `map_id` and drifting independently of `odom`. Feeding that to a
costmap means two odometry estimates fighting, and every reset teleports the
map.

The fix is on the server, not here: return the per-frame points in the
*camera* frame (CUT3R's `pts3d_in_self_view`, skipping the pose transform) and
let the existing `head_link → base_footprint → odom → map` chain place them,
with the costmap accumulating as it already does for the LiDAR. Until that
option exists, this node is useful for inspecting the reconstruction in RViz
against a static transform, not for navigation.

`map_id` changes are logged as a warning for the same reason: the world frame
restarts with them, so anything accumulated against the previous one is void.

## Running

This node is one step of a longer sequence. **The full T1 startup — the cluster
job, the tunnel, the robot, and what to watch — is in
`mecanumbot_custom_nav2/README.md` under "Starting T1".** It is written there
because that package owns the phase; what follows is only this node's own part.

One-time setup on the robot:

```bash
scp -P 10113 csengehubay@nipg36.inf.elte.hu:~/mecanumbot_repos/RoboCamStreamProcessing/link/robocam_client.py ~/
pip3 install pyzmq
```

Then, with the tunnel up (see `RoboCamStreamProcessing/link/README.md` — a node
that looks hung is usually a tunnel that is down) and a server behind it:

```bash
ros2 launch mecanumbot_deep3r deep3r.launch.py
ros2 topic hz   /mecanumbot/deep3r/points          # ~6 Hz on an RTX 3090
ros2 topic echo /mecanumbot/deep3r/map_agreement   # the server's verdict
```

`enable_map_loop` defaults to true and is what makes this a loop rather than a
one-way stream: the pose, grid and scan go up, and the verdict, the target and
the pose hint come back. Turn it off only to benchmark the transport:

```bash
ros2 launch mecanumbot_deep3r deep3r.launch.py enable_map_loop:=false
```

With it off, the server has nothing to place the cloud against, no
`MapCloudAgreement` is published, and the explorer's `CLOUD` exit criterion can
never be satisfied — so **T1 never finishes**, and nothing in the robot's logs
says why. That was the state of this package until 2026-09-08; see
`RoboCamStreamProcessing/docs/INTEGRATION.md`.

Expect roughly 6 Hz with the 512 checkpoint on an RTX 3090, less on a TITAN RTX.
`log_every` prints the point count, the server's inference time, and — when the
LiDAR is attached — the ratio between the cloud's near depth and the LiDAR's
forward range, which is the cheapest check that CUT3R's metric scale is actually
metric.

## Tests

```bash
PYTHONPATH=. python3 -m pytest test/test_cloud.py test/test_geometry.py test/test_camera_pose.py test/test_bridge.py -q
```

All run without ROS, without a server and without a GPU: the wire decoder, the
rotation helper, the neck camera model and the announcement translation are
plain numpy, and they are where a mistake produces a plausible cloud in the
wrong place rather than an error.
