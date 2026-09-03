# mecanumbot_server

ROS 2 packages for the robot's side of the **off-board compute link**: the nodes
that hand work to a server too heavy for the Orin to do itself, and translate
what comes back into ROS types.

The server itself is not in here. It lives in
[`RoboCamStreamProcessing`](https://github.com/hubaycsenge/RoboCamStreamProcessing)
on the cluster; this repository is only the robot's end of the conversation.

## Packages in this repository

| Package | Type | Runs on | Purpose |
| --- | --- | --- | --- |
| `mecanumbot_deep3r` | Python (`ament_python`) | robot (Jetson) | Off-board 3D reconstruction: streams camera JPEGs to the CUT3R server on the cluster and republishes the returned point cloud and camera pose. |

## Executables

| Executable | Package | ROS node name | Interfaces |
| --- | --- | --- | --- |
| `mecanumbot_deep3r_client_node` | `mecanumbot_deep3r` | `mecanumbot_deep3r_client_node` | Subscribes `camera/image_raw/compressed`; publishes `deep3r/points` (`PointCloud2`), `deep3r/pose` (`PoseStamped`). |

No nodes are implemented at the repository root — it only groups the package.

## What belongs here

A package belongs in `mecanumbot_server` when it does **no perception and no
control of its own** and its whole job is moving data across the robot ↔ cluster
boundary: a remote detector, a VLM, a planner, a reconstruction model. The point
of the split is that the set of things which stop working when the link is down
is a single directory rather than a memory.

The packages here run *on the robot*, which is why they are not in
`mecanumbot_remote` — that repository is now strictly what runs off the robot
(operator PC tools and the simulator). And they are not in `mecanumbot`, the
onboard Jetson stack, because nothing in the onboard stack may depend on a
network link to a machine outside the lab.

This repository previously existed, was retired into `mecanumbot_remote`, and
has been split back out for that reason.

## The link is not a detail

The robot is on the lab WiFi behind NAT; the server is on the cluster. Measured
2026-08-28: **the robot cannot reach the server's address directly.** The node
therefore talks to `tcp://127.0.0.1:<port>` and depends on a tunnel the robot
dials out itself, installed as a systemd unit from
`RoboCamStreamProcessing/link/`. Read that directory's README before debugging
anything here — a node that looks hung is usually a tunnel that is down.

## Dependencies

`mecanumbot_deep3r` depends on `rclpy`, `sensor_msgs` and `geometry_msgs`, and
needs `pyzmq`, which is **not** declared in its `package.xml` because `rosdep`
will not install it here (the same under-declaration the rest of the workspace
has — see the root `CLAUDE.md`). Install it by hand on the robot:

```bash
pip3 install pyzmq
```

The reconstruction model itself is not a dependency of anything in this repo. It
runs on the cluster; nothing here imports torch.

## Build

```bash
cd ~/Documents/mecanumbot_ws
colcon build --symlink-install --packages-select mecanumbot_deep3r
source install/setup.bash
```

## Quick run example

```bash
# on the robot, tunnel up first
ros2 launch mecanumbot_deep3r deep3r.launch.py
ros2 topic hz /mecanumbot/deep3r/points
```

`mecanumbot_deep3r` subscribes to `mecanumbot_camera_stream`'s compressed topic
rather than opening the camera itself, so the onboard camera stack has to be
running.

## Tests

```bash
cd mecanumbot_deep3r
PYTHONPATH=. python3 -m pytest test/test_cloud.py test/test_geometry.py -q
```

Both run without ROS, without a server and without a GPU.

## Repository structure

| Path | Function |
| --- | --- |
| `mecanumbot_deep3r/` | CUT3R client node, its launch file, and `config/deep3r.yaml`. |
| `LICENSE` | Apache-2.0. |

See `mecanumbot_deep3r/README.md` for the full parameter table, the timestamp
correlation, and why the returned cloud is not yet connected to TF.
