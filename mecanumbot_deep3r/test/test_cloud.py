"""
Tests for the wire-format decoder, with no ROS and no server.

The cloud arrives base64'd, quantised and rescaled against its own extent, so
a mistake here does not raise -- it produces a plausible cloud in the wrong
place, which is the failure mode a costmap is least able to survive and a
human is least likely to spot.
"""

import base64

import numpy as np
import pytest

from mecanumbot_deep3r import cloud as C


def encode(points, colors=None):
    """Quantise exactly the way the server does, so the tests exercise its format."""
    points = np.asarray(points, dtype=np.float64)
    origin = points.min(axis=0)
    extent = float(np.max(points.max(axis=0) - origin))
    scale = extent / 65535.0 if extent > 0 else 1.0
    q = np.clip((points - origin) / scale, 0, 65535).astype("<u2")
    out = {
        "n_points": int(points.shape[0]),
        "encoding": C.ENCODING,
        "origin": origin.tolist(),
        "scale": scale,
        "xyz_u16": base64.b64encode(q.tobytes()).decode("ascii"),
        "rgb_u8": base64.b64encode(np.asarray(colors, np.uint8).tobytes()).decode("ascii")
                  if colors is not None else "",
    }
    return out


def test_decode_round_trips_within_quantisation_error():
    pts = np.random.default_rng(0).uniform(-5, 5, size=(200, 3))

    back, colors = C.decode(encode(pts))

    assert back.dtype == np.float32
    assert colors is None
    # 16 bits over a 10 m extent is a fifth of a millimetre.
    assert np.abs(back - pts).max() < 1e-3


def test_decode_returns_colors_when_present():
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    rgb = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)

    _, colors = C.decode(encode(pts, rgb))

    assert colors.tolist() == rgb.tolist()


def test_an_empty_cloud_is_a_normal_frame():
    """Everything filtered out is a reading, not an error."""
    points, colors = C.decode({"n_points": 0})

    assert points.shape == (0, 3)
    assert colors is None


def test_an_unknown_encoding_is_refused():
    """A future format must not be silently read as this one."""
    cloud = encode(np.zeros((2, 3)))
    cloud["encoding"] = "u32le-xyz"

    with pytest.raises(C.CloudFormatError, match="unknown cloud encoding"):
        C.decode(cloud)


def test_a_truncated_payload_is_refused():
    cloud = encode(np.random.default_rng(1).uniform(0, 1, size=(10, 3)))
    cloud["n_points"] = 20

    with pytest.raises(C.CloudFormatError, match="claims 20 points"):
        C.decode(cloud)


def test_mismatched_colours_are_dropped_not_fatal():
    """Geometry is what navigation needs; losing a cloud over its colours is worse."""
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    cloud = encode(pts, np.array([[1, 2, 3], [4, 5, 6]], np.uint8))
    cloud["rgb_u8"] = base64.b64encode(bytes([1, 2, 3])).decode("ascii")

    points, colors = C.decode(cloud)

    assert points.shape == (2, 3)
    assert colors is None


def test_has_cloud_rejects_a_skipped_result():
    """A skipped frame still replies -- the client counts replies -- but has nothing."""
    assert C.has_cloud({"cloud": {}}) is True
    assert C.has_cloud({"cloud": {}, "status": "skipped"}) is False
    assert C.has_cloud({}) is False
    assert C.has_cloud(None) is False


def test_to_xyzrgb_lays_points_out_for_pointcloud2():
    pts = np.array([[1.0, 2.0, 3.0]], np.float32)
    rgb = np.array([[255, 128, 64]], np.uint8)

    packed = C.to_xyzrgb(pts, rgb)

    assert packed.shape == (1, 4)
    assert packed[0, :3].tolist() == [1.0, 2.0, 3.0]
    # RViz reads the fourth float's *bits* as 0x00RRGGBB, so the round trip
    # has to go through the bit pattern rather than the value.
    assert packed[0, 3:4].view(np.uint32)[0] == (255 << 16) | (128 << 8) | 64


def test_to_xyzrgb_without_colours_omits_the_channel_entirely():
    """
    It used to leave a zeroed fourth column, which is solid black.

    An rgb channel of zeros is not "no colour", it is the colour black, and
    RViz colouring by RGB8 has nothing else to fall back to.  Three columns
    with no rgb field lets it pick a transformer that says something.
    """
    packed = C.to_xyzrgb(np.array([[1.0, 2.0, 3.0]], np.float32), None)
    assert packed.shape == (1, 3)
    assert packed[0].tolist() == [1.0, 2.0, 3.0]


class TestLayout:
    """
    What the ``PointCloud2`` says about itself, and that the bytes agree.

    The server ships a ``pc2`` block so this end does not keep its own copy of
    the layout.  It kept one anyway until 2026-09-17, and the cost was a
    colourless cloud published as four fields with a zeroed ``rgb`` -- solid
    black, with no other channel for RViz to colour by.
    """

    def with_colors(self):
        return {"pc2": {"point_step": 16, "fields": [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            {"name": "rgb", "offset": 12, "datatype": 6, "count": 1},
        ]}}

    def without_colors(self):
        block = self.with_colors()
        block["pc2"]["fields"] = block["pc2"]["fields"][:3]
        block["pc2"]["point_step"] = 12
        return block

    def test_a_coloured_cloud_carries_the_rgb_field(self):
        fields, step = C.layout(self.with_colors(), True)
        assert [f["name"] for f in fields] == ["x", "y", "z", "rgb"]
        assert step == 16

    def test_the_rgb_field_is_published_as_float32(self):
        """The one thing not taken from the server: pcl::PointXYZRGB wants it."""
        fields, _ = C.layout(self.with_colors(), True)
        assert fields[-1]["datatype"] == C.FLOAT32

    def test_a_colourless_cloud_drops_the_field_rather_than_blacking_it(self):
        fields, step = C.layout(self.without_colors(), False)
        assert [f["name"] for f in fields] == ["x", "y", "z"]
        assert step == 12

    def test_the_bytes_decide_when_the_declaration_disagrees(self):
        """A dropped colour array must not leave an rgb field over zeros."""
        fields, step = C.layout(self.with_colors(), False)
        assert [f["name"] for f in fields] == ["x", "y", "z"]
        assert step == 12

    def test_an_undeclared_layout_falls_back(self):
        """An older server sends no pc2 block; the cloud is still publishable."""
        fields, step = C.layout({}, True)
        assert [f["name"] for f in fields] == ["x", "y", "z", "rgb"]
        assert step == 16

    def test_the_step_matches_the_bytes_to_xyzrgb_produces(self):
        """The pair that has to hold, or every point is strided off its own end."""
        points = np.zeros((7, 3), np.float32)
        for colors in (np.zeros((7, 3), np.uint8), None):
            declared = self.with_colors() if colors is not None else self.without_colors()
            _, step = C.layout(declared, colors is not None)
            packed = C.to_xyzrgb(points, colors)
            assert len(packed.tobytes()) == step * 7

    def test_an_unusable_declaration_is_refused_not_guessed(self):
        with pytest.raises(C.CloudFormatError):
            C.layout({"pc2": {"fields": [{"name": "x"}]}}, True)


class TestColourNote:
    """
    Why a cloud arrived grey, in words, instead of silently.

    A dropped colour array and a genuinely grey picture look identical in
    RViz, so the one the robot *can* tell apart has to say so itself.
    """

    def test_a_good_cloud_has_nothing_to_say(self):
        points = np.array([[1.0, 2.0, 3.0]])
        colors = np.array([[10, 20, 30]], np.uint8)
        assert C.colour_note(encode(points, colors)) is None

    def test_an_empty_cloud_has_nothing_to_say(self):
        assert C.colour_note({"n_points": 0}) is None

    def test_no_colour_at_all_names_the_server_option(self):
        note = C.colour_note(encode(np.array([[1.0, 2.0, 3.0]])))
        assert "colors" in note

    def test_the_wrong_amount_of_colour_gives_both_counts(self):
        cloud = encode(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                       np.array([[1, 2, 3], [4, 5, 6]], np.uint8))
        cloud["n_points"] = 5                    # as if the arrays disagreed
        note = C.colour_note(cloud)
        assert "6 colour bytes" in note and "5 points" in note

    def test_undecodable_colour_is_reported_not_raised(self):
        assert "base64" in C.colour_note({"n_points": 2, "rgb_u8": "!!!not base64!!!"})
