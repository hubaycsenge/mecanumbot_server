"""
Turning the server's announcements into things the robot's nodes act on.

The server sends four things the robot never asks for: ``map_update`` (cells to
merge), ``pose_hint`` (a correction offered to SLAM), ``agreement`` (how the
cloud and the grid compare, and where they do not) and ``found`` (the target).
This module is the translation, and it is deliberately **free of ROS**: it takes
header dicts and returns plain data, so the part where a wrong index or a
swallowed ``None`` produces a plausible wrong answer can be tested without a
graph, a server or a GPU.

Two rules run through it.

**Parallel arrays are checked, not trusted.** ``MapCloudAgreement`` carries six
arrays the robot indexes with one subscript. If they ever disagree in length,
the robot reads a region's kind against another region's height — a table
reported at the position of a patch of unobserved floor — and nothing downstream
can tell. So a length mismatch raises here rather than being padded or zipped
short.

**An unmeasurable height stays unmeasurable.** ``null`` on the wire becomes
``None``, never ``0.0`` and never ``nan``. Zero is the floor, which the robot
drives over; ``nan`` fails every comparison silently, including the one that
decides whether it drives into the thing. ``None`` is the only value that makes
``mecanumbot_map_agreement`` treat the region as blocking, which is what an
unknown height must be.
"""

from __future__ import annotations

#: Kinds of disagreement, matching ``robocam.regions`` and the message comment.
KIND_CLOUD_ONLY = "cloud_only"
KIND_MAP_ONLY = "map_only"
KIND_UNOBSERVED = "unobserved"
KIND_DISAGREEMENT = "disagreement"
KINDS = (KIND_CLOUD_ONLY, KIND_MAP_ONLY, KIND_UNOBSERVED, KIND_DISAGREEMENT)

#: What ``found.basis`` says about which topic the answer belongs on.  These are
#: the seek tree's two inputs and the distinction its whole two-branch search
#: rests on: a memory is where the object *was*, a live sighting is where it
#: *is*.  Merging them would leave the tree unable to tell them apart.
BASIS_LIVE = "live"
BASIS_MEMORY = "memory"
BASIS_ABSENT = "absent"


class BridgeError(ValueError):
    """A malformed announcement.  Raised rather than repaired."""


def _floats(header, key):
    values = header.get(key) or []
    if not isinstance(values, (list, tuple)):
        raise BridgeError(f"{key} is {type(values).__name__}, expected a list")
    return [float(v) for v in values]


def agreement_regions(header):
    """
    Validate and unpack the six parallel arrays of an ``agreement`` header.

    Returns a list of dicts — ``x``, ``y``, ``score``, ``kind``, ``height``,
    ``radius`` — in the order they arrived, which is the server's ranking.

    ``height`` is ``None`` where the server could not measure one. Every other
    field is required: a region without a position is not a place, and a region
    without a kind is not a decision.
    """
    xs = _floats(header, "uncertain_x")
    ys = _floats(header, "uncertain_y")
    scores = _floats(header, "uncertain_scores")
    radii = _floats(header, "uncertain_radii")
    kinds = [str(k) for k in (header.get("uncertain_kinds") or [])]
    raw_heights = header.get("uncertain_heights") or []

    lengths = {len(xs), len(ys), len(scores), len(radii), len(kinds), len(raw_heights)}
    if len(lengths) > 1:
        raise BridgeError(
            "agreement arrays are not parallel: "
            f"x={len(xs)} y={len(ys)} scores={len(scores)} kinds={len(kinds)} "
            f"heights={len(raw_heights)} radii={len(radii)}"
        )

    heights = [None if h is None else float(h) for h in raw_heights]
    for kind in kinds:
        if kind not in KINDS:
            raise BridgeError(f"unknown region kind {kind!r}")

    return [
        {"x": x, "y": y, "score": s, "kind": k, "height": h, "radius": r}
        for x, y, s, k, h, r in zip(xs, ys, scores, kinds, heights, radii)
    ]


def is_stale(header, current_map_id):
    """
    Say whether an announcement names a map the robot is no longer on.

    ``map_id`` is **the robot's SLAM map**, echoed back by the server — not
    CUT3R's reconstruction session, which travels separately as
    ``cloud_map_id``. The two invalidate different things and the names have
    been confused before: a coordinate carried across a robot ``map_id`` change
    names a place in a map that no longer exists, while a ``cloud_map_id``
    change means the reconstruction those coordinates were derived from has
    restarted.  This function is about the first.

    An announcement with no ``map_id`` at all is *not* stale: a server that did
    not say cannot be assumed to disagree, and the map loop has to survive a
    session that started before SLAM published its first map.
    """
    if not current_map_id:
        return False
    announced = header.get("map_id")
    if not announced:
        return False
    return str(announced) != str(current_map_id)


class MapIdentity:
    """
    The robot's SLAM map identity, which changes when the map frame does.

    It used to be the grid's geometry -- ``<width>x<height>@<resolution>+<origin>``
    -- on the reasoning that size and origin change exactly when slam_toolbox
    re-rasterises.  They do not: slam_toolbox sizes the grid to the bounding box
    of every scan so far, so during T1 the width, height and origin change on
    almost every ``/map`` update, whenever the robot sees past the previous box.
    The map frame does not move when that happens -- a point at (3.2, 1.4) m is
    still at (3.2, 1.4) m -- but every such update was a new ``map_id``, and
    three things are keyed on it:

    * the server drops a verdict computed against a grid it has since replaced,
      and this node drops one computed against a grid it has since received, so
      most verdicts never reached the robot;
    * the ones that did carried a new id each time, and
      ``mecanumbot_map_agreement`` clears every accumulated region on a new id,
      so no keepout outlived one ``/map`` update;
    * and T1's ``CLOUD`` exit criterion, starved of verdicts, could not be met.

    What does move the map frame is SLAM starting over, so that is what bumps
    the id: a different resolution or frame, or the observed area collapsing to
    under ``restart_fraction`` of what it was.  A loop closure redraws the map
    with a few percent fewer cells and is not a restart; a restarted
    slam_toolbox begins again from a single scan.

    The id is deterministic -- no process token in it -- so a reconnect, or this
    node restarting while SLAM carries on, still names the same map, and the
    server's T1 sightings stay valid into T2.
    """

    def __init__(self, restart_fraction=0.5):
        self.restart_fraction = float(restart_fraction)
        self.session = 0
        self._resolution = None
        self._frame = None
        self._known_cells = None

    def observe(self, resolution, frame, known_cells):
        """Take one grid's resolution, frame and observed-cell count; return the id."""
        resolution = round(float(resolution), 4)
        frame = str(frame or "map")
        known_cells = int(known_cells)
        if self._known_cells is not None and (
                resolution != self._resolution
                or frame != self._frame
                or known_cells < self._known_cells * self.restart_fraction):
            self.session += 1
        self._resolution = resolution
        self._frame = frame
        self._known_cells = known_cells
        return self.map_id

    @property
    def map_id(self):
        """Return the current id, or ``""`` before any grid has been seen."""
        if self._resolution is None:
            return ""
        return f"{self._frame}@{self._resolution:.4f}#{self.session}"


def cloud_map_id(header):
    """
    CUT3R's reconstruction session for this announcement, or ``None``.

    Kept distinct from :func:`is_stale`'s ``map_id`` on purpose — see this
    module's docstring and ``docs/INTEGRATION.md`` in RoboCamStreamProcessing.
    A change here voids anything derived from the cloud's geometry; it does not
    invalidate the robot's map.
    """
    value = header.get("cloud_map_id")
    return None if value is None else int(value)


def found_topic(header):
    """
    Which of the seek tree's two inputs a ``found`` announcement belongs on.

    The server's decision stage is the only object detector in this system, so
    it produces both of the tree's inputs — but they are not the same claim and
    must not arrive on the same topic:

    ``live``    the stage is looking at the target in this frame.  Perception:
                briefly true, many per second, and what the monitor branch
                interrupts the search on.  Goes to ``seek/detections``.
    ``memory``  the target was seen earlier, usually during T1 before it was
                named.  One hypothesis that stays put until replaced, and what
                the search rings are centred on.  Goes to ``seek/target``.
    ``absent``  the stage looked and it is not here.  Goes to neither: an empty
                array on either topic would be read as "no detection this
                frame", which is what a blank wall also looks like, and the
                tree would lose the difference between "not seen" and "looked
                for and not there".

    Returns ``None`` for ``absent`` and for ``found: false``.
    """
    if not header.get("found", False):
        return None
    basis = str(header.get("basis", BASIS_LIVE))
    if basis == BASIS_MEMORY:
        return "target"
    if basis == BASIS_LIVE:
        return "detections"
    return None


def found_hypothesis(header):
    """
    Reduce a ``found`` header to the fields a ``Detection3DArray`` carries.

    ``z`` travels as the position's z and is the whole reason this system has a
    reconstruction: it is what ``CheckObjectGraspable`` reads, and what turns
    "something is at (x, y) and I cannot reach it" into "it is on a table".

    The server's own ``reachable`` verdict is carried alongside but is **not**
    what decides the grasp. The tree judges reachability against the robot's own
    geometry, and having two authorities on one question is how the two ends
    came to disagree about the grasp band in the first place. The server's
    verdict is advisory here, for the log and for a future consumer that wants
    to skip an approach it knows will fail.
    """
    approach = header.get("approach") or {}
    reach = header.get("reach") or {}
    return {
        "class_id": str(header.get("target", "")),
        "score": float(header.get("confidence", 0.0)),
        "x": float(header.get("x", 0.0)),
        "y": float(header.get("y", 0.0)),
        "z": float(header.get("z", 0.0)),
        "yaw": float(header.get("yaw", 0.0)),
        "basis": str(header.get("basis", BASIS_LIVE)),
        "age_s": float(header.get("age_s", 0.0)),
        "approach_x": float(approach.get("x", header.get("x", 0.0))),
        "approach_y": float(approach.get("y", header.get("y", 0.0))),
        "approach_yaw": float(approach.get("yaw", header.get("yaw", 0.0))),
        # Three-valued and left that way: None means the height could not be
        # measured, which is a different instruction from False.
        "reachable": header.get("reachable"),
        "reach_verdict": str(reach.get("verdict", "")),
    }


def pose_hint_is_usable(header, min_confidence=0.0, min_inliers=0):
    """
    Whether a ``pose_hint`` is worth passing on at all.

    The hint is advisory in the protocol and stays advisory here. This is not
    the decision to apply it — nothing in this workspace applies one, because
    slam_toolbox owns the robot's pose and has a graph the server cannot see.
    It is the filter on whether the hint is even worth logging as a correction
    the operator might want to know about: a hint from a handful of inliers is
    the reconstruction agreeing with one corner of the room.
    """
    if not header.get("advisory", True):
        # A server claiming its hint is not advisory is a server speaking a
        # protocol this robot does not implement.  Refuse rather than obey.
        raise BridgeError("pose_hint arrived with advisory=false; refusing it")
    return (float(header.get("confidence", 0.0)) >= float(min_confidence)
            and int(header.get("inliers", 0)) >= int(min_inliers))


#: Constructor keywords this node hands the client. Each arrived with protocol
#: 2; a client older than that has none of them.
REQUIRED_CLIENT_KWARGS = (
    "map_every_s", "on_map_update", "on_pose_hint", "on_found", "on_agreement",
    "run_id",
)


def client_takes_frame_pose(module):
    """
    Say whether a deployed client can attach a pose to a frame.

    Checked rather than required, unlike the map-loop keywords: a client
    without it still runs the whole loop and loses only the per-frame pose,
    which the node says once at startup.  Refusing to start over it would take
    the robot's T1 down for an accuracy problem.
    """
    import inspect

    try:
        params = inspect.signature(module.RoboCamClient._send_frame).parameters
    except (AttributeError, ValueError, TypeError):
        return False
    return "pose" in params


def check_client_api(module, path):
    """
    Refuse a deployed client older than this node, with a usable message.

    This pair is the one place in the workspace where two repositories have to
    be updated together and only one of them is a git dependency: the node comes
    from `mecanumbot_server` via `git pull`, and `robocam_client.py` is a single
    file `scp`-ed from `RoboCamStreamProcessing/link`. Updating the workspace
    without redeploying the file is therefore the normal mistake, not an exotic
    one -- and left alone it surfaces as `TypeError: __init__() got an
    unexpected keyword argument 'map_every_s'` from inside a constructor, which
    names the symptom and not the cause.
    """
    import inspect

    try:
        params = inspect.signature(module.RoboCamClient.__init__).parameters
    except (AttributeError, ValueError, TypeError):  # pragma: no cover
        return          # not introspectable; let the real call fail normally

    missing = [k for k in REQUIRED_CLIENT_KWARGS if k not in params]
    if not missing:
        return
    import os

    header = (
        f"the robocam_client.py at {path} is older than this node: it does not "
        f"accept {', '.join(missing)}.\n"
    )
    real = os.path.realpath(path)
    if real != os.path.abspath(path):
        # A symlink into a checkout is only as new as that checkout: linking
        # the file does not update it, pulling the repository it lives in does.
        fix = (
            f"It is a link to {real}, so that checkout is behind. Update it:\n"
            f"  git -C {os.path.dirname(real)} pull\n"
        )
    else:
        fix = (
            "That file is deployed by scp, not by git, so a `git pull` of this "
            "workspace updates the node and leaves the client behind. "
            "Redeploy it:\n"
            "  scp <host>:.../RoboCamStreamProcessing/link/robocam_client.py ~/\n"
        )
    raise RuntimeError(
        header + fix
        + "Or run without the map loop until you do:\n"
        "  ros2 launch mecanumbot_deep3r deep3r.launch.py enable_map_loop:=false"
    )
