"""
Decode the point cloud the server sends back.

Pure numpy: no ROS, no ZeroMQ, no server.  The wire format is small enough to
describe in one place and awkward enough (base64, quantised, per-cloud origin)
that getting it wrong yields a plausible-looking cloud in the wrong place, so
it is worth being able to test it on its own.

The server quantises each cloud against its *own* extent rather than a fixed
scale, so ``origin`` and ``scale`` differ from frame to frame and both are
needed to reconstruct metres:

    p = origin + scale * u        u is little-endian uint16, xyz interleaved
"""

import base64

import numpy as np

#: What the server writes in ``cloud["encoding"]``.  Checked rather than
#: assumed: a future encoding would otherwise be silently misread as this one.
ENCODING = "u16le-xyz"


class CloudFormatError(ValueError):
    """The payload is not a cloud this decoder understands."""


def has_cloud(result):
    """
    Say whether a result carries a cloud at all.

    A result with ``status`` is one the server chose not to run the model on
    (``every_n``), and it still has to be counted -- the client sizes its
    in-flight window from replies -- but there is nothing to publish.
    """
    return isinstance(result, dict) and "cloud" in result and "status" not in result


def decode(cloud):
    """
    Return ``(points, colors)`` for one cloud.

    ``points`` is (N, 3) float32 in metres; ``colors`` is (N, 3) uint8 or None
    when the server was configured without them.  An empty cloud -- everything
    filtered out, which is a normal frame rather than an error -- comes back as
    a (0, 3) array rather than None, so callers do not need a second branch.
    """
    if not isinstance(cloud, dict):
        raise CloudFormatError(f"expected a dict, got {type(cloud).__name__}")

    n = int(cloud.get("n_points", 0))
    if n == 0:
        return np.zeros((0, 3), np.float32), None

    encoding = cloud.get("encoding")
    if encoding != ENCODING:
        raise CloudFormatError(
            f"unknown cloud encoding {encoding!r}; this decoder handles {ENCODING!r}"
        )

    raw = np.frombuffer(base64.b64decode(cloud["xyz_u16"]), "<u2")
    if raw.size != n * 3:
        raise CloudFormatError(
            f"cloud claims {n} points but carries {raw.size / 3:.1f}"
        )

    origin = np.asarray(cloud["origin"], dtype=np.float64)
    scale = float(cloud["scale"])
    points = (origin + scale * raw.reshape(-1, 3)).astype(np.float32)

    colors = None
    packed = cloud.get("rgb_u8") or ""
    if packed:
        rgb = np.frombuffer(base64.b64decode(packed), np.uint8)
        if rgb.size == n * 3:
            colors = rgb.reshape(-1, 3)
        # A colour array of the wrong length is dropped rather than raised on:
        # the geometry is what navigation needs, and losing a whole cloud over
        # its colours would be the wrong trade.

    return points, colors


#: ``sensor_msgs/PointField`` datatype numbers, so this module can talk about a
#: layout without importing ROS.
FLOAT32 = 7
UINT32 = 6

#: The layout to publish when the server declares none -- an older server, or a
#: reply whose ``cloud`` block predates ``pc2``.
_FALLBACK_FIELDS = (
    {"name": "x", "offset": 0, "datatype": FLOAT32, "count": 1},
    {"name": "y", "offset": 4, "datatype": FLOAT32, "count": 1},
    {"name": "z", "offset": 8, "datatype": FLOAT32, "count": 1},
)
_RGB_FIELD = {"name": "rgb", "offset": 12, "datatype": FLOAT32, "count": 1}


def layout(cloud, have_colors):
    """
    Return ``(fields, point_step)`` for one cloud's ``PointCloud2``.

    Taken from the server's own ``pc2`` block where it sent one.  It ships that
    block precisely so this end does not keep a second copy of the layout --
    "exactly the kind of duplicated constant that survives a change at this end
    and produces a cloud read as garbage at the other", as ``_pack_cloud`` puts
    it -- and until 2026-09-17 this end kept one anyway: four fields and a
    ``point_step`` of 16, whatever the server said.

    What that cost was a **colourless cloud published as a black one**.  A
    server configured with ``colors: false`` sends three fields and a
    ``point_step`` of 12; the hardcoded layout appended an ``rgb`` field over
    the zeros that :func:`to_xyzrgb` leaves there, so every point arrived
    solid black and RViz had no other channel to fall back to.  ``have_colors``
    is therefore what decides whether the field exists, not what the server
    declared: the two disagree exactly when a colour array was dropped for
    being the wrong length, and the bytes win.

    The one thing deliberately **not** taken from the server is the ``rgb``
    field's datatype.  The server declares ``UINT32``; this publishes
    ``FLOAT32``, which is the same four bytes read the same way and is the
    convention ``pcl::PointXYZRGB`` requires -- so a cloud that reaches a Nav2
    costmap layer through PCL is read rather than rejected.  RViz accepts
    either.
    """
    fields = None
    declared = cloud.get("pc2") if isinstance(cloud, dict) else None
    if isinstance(declared, dict):
        raw = declared.get("fields")
        if isinstance(raw, (list, tuple)) and raw:
            try:
                fields = [
                    {"name": str(f["name"]), "offset": int(f["offset"]),
                     "datatype": int(f["datatype"]), "count": int(f.get("count", 1))}
                    for f in raw
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise CloudFormatError(f"unusable pc2.fields: {exc}") from exc

    if fields is None:
        fields = [dict(f) for f in _FALLBACK_FIELDS]

    # The bytes decide, not the declaration.  Drop a colour field we have no
    # colours for rather than publish the zeros as black; add one the server
    # forgot to declare but sent the bytes for.
    fields = [f for f in fields if f["name"] != "rgb"]
    if have_colors:
        fields.append(dict(_RGB_FIELD, offset=max(f["offset"] + 4 for f in fields)))
    for field in fields:
        if field["name"] == "rgb":
            field["datatype"] = FLOAT32

    # Derived, never taken from ``pc2.point_step``: :func:`to_xyzrgb` packs
    # exactly these fields and nothing else, so a declared step that disagreed
    # -- 16 from a server that sent colours, for a reply whose colours were
    # dropped -- would stride the buffer past the end of every point.
    return fields, max(f["offset"] for f in fields) + 4


def to_xyzrgb(points, colors):
    """
    Pack into the interleaved xyz+rgb buffer PointCloud2 wants.

    RViz and the Nav2 costmaps read ``rgb`` as a float32 whose bits are
    ``0x00RRGGBB``; that is a convention rather than anything numeric, hence
    the view rather than a cast.

    Returns (N, 3) when there are no colours, which is what :func:`layout`
    declares for that case.  The two are a pair: the fields and the bytes have
    to agree about how wide a point is.
    """
    n = points.shape[0]
    if colors is None:
        # Three columns, not four with a zeroed fourth: an all-zero ``rgb``
        # column is solid black, and :func:`layout` drops the field to match.
        return np.ascontiguousarray(points, dtype=np.float32).reshape(n, 3)
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, :3] = points
    packed = (colors[:, 0].astype(np.uint32) << 16
              | colors[:, 1].astype(np.uint32) << 8
              | colors[:, 2].astype(np.uint32))
    out[:, 3] = packed.view(np.float32)
    return out
