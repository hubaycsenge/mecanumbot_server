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
| `/camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Forwarded to the server **as the JPEG it already is** — no decode/re-encode on the robot. |

### Publishers

| Topic | Data type | Function |
| --- | --- | --- |
| `deep3r/points` | `sensor_msgs/msg/PointCloud2` | The returned cloud: `xyz` + packed `rgb`, metres. |
| `deep3r/pose` | `geometry_msgs/msg/PoseStamped` | CUT3R's camera pose for that frame, same frame as the cloud. Disable with `publish_pose`. |
| `/tf` | `map -> deep3r_world` | Where CUT3R's world frame sits in the map, per cloud. Disable with `publish_world_tf`. |

### Parameters

| Parameter | Default | Function |
| --- | --- | --- |
| `server` | `tcp://127.0.0.1:5555` | Local end of the forward tunnel — **not** the server's own address. |
| `client_path` | `~/robocam_client.py` | The deployed standalone client; a directory is also accepted. |
| `camera_topic` | `/camera/image_raw/compressed` | Where frames come from. **Absolute**: the publisher is not namespaced and this node is. A relative name resolves to `/mecanumbot/camera/...`, which nothing publishes, and the node then sends no frames and logs no error. |
| `cloud_topic` | `deep3r/points` | Point cloud output. |
| `pose_topic` | `deep3r/pose` | Camera pose output. |
| `publish_pose` | `true` | Publish the pose alongside the cloud. |
| `frame_id` | `deep3r_world` | Frame the cloud and pose are stamped in: CUT3R's world frame. |
| `publish_world_tf` | `true` | Broadcast `map_frame -> frame_id` per cloud. Needs the map loop and both `send_frame_pose` and `send_camera_pose`. See below. |
| `advertised_width` / `_height` / `_fps` | `1280` / `720` / `15.0` | Metadata the client announces. The server reports what it actually decoded, so these need not be exact. |
| `max_inflight` | `2` | Frames awaiting a reply. |
| `queue_depth` | `2` | Frames buffered between the subscription and the client. |
| `log_every` | `30` | Log one line every N clouds; 0 disables. |
| `run_id` | *(empty)* | Which run this is. Empty mints a fresh one at start-up, and the server wipes its reconstruction for a run it has not seen. Pass a previous id to resume; it is also a launch argument of `deep3r.launch.py`, `launch_autoslam.launch.py` and `launch_t1.launch.py`. |
| `send_frame_pose` | `true` | Attach the robot's map pose at the image's stamp to every frame. See below. |
| `odom_frame` | `mecanumbot/odom` | Fixed frame for that lookup: `odom -> base` at the stamp, `map -> odom` at its latest. |
| `pose_lookup_timeout_s` | `0.05` | How long a frame waits for odometry to reach its stamp. |
| `send_camera_pose` | `true` | Attach the camera's pose from the neck model as well. |
| `neck_topic` / `neck_stale_s` | `opencr_state` / `0.5` | Where the neck position comes from, and how far from a frame it may be. |
| `camera.pivot_x` / `_z`, `camera.lever_x` / `_z` | `0.1063` / `0.1679`, `0.022` / `0.038` | The neck pivot and the pivot-to-lens lever, in `base_link`. |
| `camera.level_ticks` | `600.0` | Servo ticks of the trees' `neck_level_pos`. |
| `camera.rad_per_tick` | `0.005061` | Tilt per servo tick. |
| `camera.pitch_at_level_deg` | `0.0` | **Unmeasured.** Lens tilt at `level_ticks`, positive up. |

A few more subscriptions come with the map loop: `/map`, `/mecanumbot/scan`, the explorer's
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

**It is not a TF lookup, although TF now agrees with it.** Until 2026-09-14,
`camera_rgb_optical_frame` in the URDF looked along the neck's own axis (to the
robot's right) at every neck angle. Also, `head_joint` read about 40° back at
level. Both are fixed in `mecanumbot` (`mecanumbot_description` URDF,
`mecanumbot_sensorproc_node`): `head_joint` is zero at `level_ticks` (600) and
positive looking up, the same convention as this model.

Checked by composing the URDF chain against `NeckCamera.extrinsic` from 200 to
860 ticks:

- **Rotation:** the two agree to within 0.1°. The remainder comes from the URDF
  writing `1.57` where it means π/2.
- **Position:** `camera_link`'s origin agrees with the model's lens to within
  0.03 mm. `camera_rgb_optical_frame` sits **14.5 mm** further along, because of
  `camera_rgb_joint`'s offset, which the model does not include. Nobody has
  measured which of the two is where the lens actually is.

The model is still used instead of TF, for three reasons:

- **The pose has to be the neck at the image's stamp.** The model reads the
  neck reading closest to that stamp directly.
- **`pitch_at_level_deg` has nowhere to go in the URDF.** TF would drop the one
  calibration this placement has.
- **The simulator still uses the old tick convention.** `mecanumbot_sim`'s
  `accessory_ticks_to_angle` is `2.618 - ticks · rad_per_tick`: the opposite
  sign and a different zero. In simulation, `head_joint` in TF does not mean
  what it means on the robot.

Since the two now agree, TF is the cross-check. If RViz shows the cloud tilted
against `camera_rgb_optical_frame`, one of them has drifted from the other.
`camera_pose.py` and `mecanumbot_sensorproc_node.NECK_LEVEL_TICKS` have to
change together, as the comment there says.

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

`mecanumbot_camera_stream`'s compressed publisher owns the camera device (the
USB webcam, `/dev/video0`) whenever the topic exists. A second process opening
the device would either fail or take it away. Consuming the compressed topic
also means the JPEG is already encoded, so frames are forwarded untouched rather
than decoded and re-encoded on the robot's CPU.

The topic is **`/camera/image_raw/compressed`, absolute**. The publisher is not
namespaced and this node is. Until 2026-09-15 `camera_topic` was relative and
resolved to `/mecanumbot/camera/image_raw/compressed`, which nothing publishes.
The node then ran and reached the server, but sent no frames and logged no
error, so T1 could not finish.

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

## The world frame in TF: `map -> deep3r_world`

The cloud arrives in CUT3R's own world frame, anchored on the first frame of
each reconstruction session with an origin and orientation nobody chose. The
node broadcasts where that frame sits in the map, **once per cloud, stamped
with the cloud's own stamp**:

```text
T_map_world = T_map_base · T_base_optical · (pose_c2w)⁻¹
```

`T_map_base` and `T_base_optical` are the pose this node attached to the frame
(see above), kept alongside the hand-over record. `pose_c2w` is the camera pose
the server returned with the cloud. This is the same product over the same
inputs as `map_from_cloud_matrix` in the server's `robocam/compare.py`,
including its planar base (yaw only). The cloud in RViz therefore lands exactly
where the comparison put it, and `agreement` describes what you see. Checked
against the server's function, including its frame-pose decoder: identical to
1e-15 over 200 random frames. `world_frame.py` holds the algebra and
`test/test_world_frame.py` tests it.

A frame gets **no** transform unless the result's `pose_source` is `frame` and
the frame carried a camera. Otherwise the server placed the cloud with its
stream pose or its configured mount, which this node never saw. TF
interpolates between the neighbouring estimates instead.

Things to know before relying on it:

- **It is not smoothed, on purpose.** Each estimate carries that frame's
  odometry error, neck-model error and CUT3R pose drift. Averaging them would
  place clouds somewhere the comparison did not. The `log_every` line reports
  the **spread** instead: how far the current estimate is from the first one of
  this reconstruction session, in metres and degrees. A correct placement keeps
  it small. A spread that keeps growing is the same fault as an `agreement`
  near zero, seen from the other side.
- **It is rigid.** A reconstruction whose metric scale is off (see the
  lidar/cloud ratio in the same log line) is not rescaled. Its clouds come out
  the right shape at the wrong size around the camera.
- **A reconstruction reset moves the frame.** `cloud_map_id` changes are logged
  as a warning, and the spread's reference restarts with them. Anything
  accumulated against the previous world frame is void.
- **It is display and inspection, not navigation.** The per-frame cloud (≤4000
  points in 5 cm voxels) feeds nothing on the robot. The height decision the
  costmap needs comes from the server's agreement regions through
  `mecanumbot_map_agreement` in `mecanumbot_custom_nav2`.

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

**In T1 you do not launch this node yourself.** `mecanumbot_autoslam`'s
`launch_autoslam.launch.py` includes `deep3r.launch.py` together with the camera
publisher, and `launch_t1.launch.py` and the web GUI's Autoslam row both go
through it. So, with the tunnel up (see `RoboCamStreamProcessing/link/README.md`
— a node that looks hung is usually a tunnel that is down) and a server behind
it:

```bash
ros2 launch mecanumbot_autoslam launch_t1.launch.py
ros2 topic hz   /mecanumbot/deep3r/points          # ~6 Hz on an RTX 3090
ros2 topic echo /mecanumbot/deep3r/map_agreement   # the server's verdict
```

On its own, for a transport check or T2, it needs something publishing
`/camera/image_raw/compressed` first. Nothing else starts the camera: the base
launch stopped, and so did perception. The camera is the robot's USB webcam:

```bash
ros2 launch mecanumbot_camera_stream camera_compressed.launch.py width:=1280 height:=720
ros2 launch mecanumbot_deep3r deep3r.launch.py
```

Do not start a second client while one is running, for example this command
next to a T1 launch. It is a second session with its own `run_id`, and the
server wipes its reconstruction every time the `run_id` it sees changes.

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
PYTHONPATH=. python3 -m pytest test/test_cloud.py test/test_geometry.py test/test_camera_pose.py test/test_bridge.py test/test_world_frame.py -q
```

All run without ROS, without a server and without a GPU: the wire decoder, the
rotation helper, the neck camera model, the world frame's placement and the
announcement translation are
plain numpy, and they are where a mistake produces a plausible cloud in the
wrong place rather than an error.
