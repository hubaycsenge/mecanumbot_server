"""
Tests for the translation between the server's announcements and ROS.

Written around the failures that produce a plausible wrong answer rather than an
error: parallel arrays read off by one, an unmeasurable height quietly becoming
a floor, and the two map identities being confused for each other.

Pure Python -- no ROS, no server, no GPU.
"""

import pytest

from mecanumbot_deep3r import bridge


def an_agreement(n=2, **overrides):
    header = {
        "type": "agreement",
        "frame": "map",
        "map_id": "40x40@0.0500+0.000,0.000",
        "cloud_map_id": 3,
        "agreement": 0.42,
        "grid_coverage": 0.8,
        "uncertain_x": [1.0, 2.0][:n],
        "uncertain_y": [3.0, 4.0][:n],
        "uncertain_scores": [1.0, 0.5][:n],
        "uncertain_kinds": ["cloud_only", "unobserved"][:n],
        "uncertain_heights": [0.74, None][:n],
        "uncertain_radii": [0.5, 1.2][:n],
    }
    header.update(overrides)
    return header


def a_found(**overrides):
    header = {
        "type": "found",
        "found": True,
        "target": "the red mug on the desk",
        "confidence": 0.8,
        "frame": "map",
        "map_id": "40x40@0.0500+0.000,0.000",
        "x": 3.2, "y": 1.4, "z": 0.74, "yaw": 0.1,
        "basis": "live",
        "age_s": 0.0,
        "reachable": False,
        "reach": {"verdict": "too_high"},
        "approach": {"x": 2.5, "y": 1.4, "yaw": 0.0},
    }
    header.update(overrides)
    return header


# --- the parallel arrays ---------------------------------------------------

def test_regions_unpack_in_order():
    regions = bridge.agreement_regions(an_agreement())
    assert [r["kind"] for r in regions] == ["cloud_only", "unobserved"]
    assert regions[0]["x"] == 1.0
    assert regions[0]["radius"] == 0.5


def test_arrays_of_different_lengths_are_refused():
    """A short zip would read one region's kind against another's height."""
    header = an_agreement()
    header["uncertain_heights"] = [0.74]        # one short
    with pytest.raises(bridge.BridgeError, match="not parallel"):
        bridge.agreement_regions(header)


def test_an_unknown_kind_is_refused():
    """The robot switches on the kind; an unhandled one silently does nothing."""
    header = an_agreement()
    header["uncertain_kinds"] = ["cloud_only", "something_new"]
    with pytest.raises(bridge.BridgeError, match="unknown region kind"):
        bridge.agreement_regions(header)


def test_an_empty_verdict_is_valid():
    """A comparison that found nothing to report is news, not a malformation."""
    header = an_agreement(n=0)
    assert bridge.agreement_regions(header) == []


def test_a_null_height_stays_none():
    """Not 0.0, which is the floor and gets driven over."""
    regions = bridge.agreement_regions(an_agreement())
    unobserved = [r for r in regions if r["kind"] == "unobserved"][0]
    assert unobserved["height"] is None


# --- the two map identities ------------------------------------------------

def test_a_verdict_for_another_map_is_stale():
    assert bridge.is_stale(an_agreement(), "some-other-map") is True


def test_a_verdict_for_the_current_map_is_not_stale():
    header = an_agreement()
    assert bridge.is_stale(header, header["map_id"]) is False


def test_nothing_is_stale_before_the_robot_has_a_map():
    """The map loop has to survive a session that starts before SLAM publishes."""
    assert bridge.is_stale(an_agreement(), "") is False


def test_a_server_that_did_not_say_is_not_assumed_to_disagree():
    header = an_agreement()
    header.pop("map_id")
    assert bridge.is_stale(header, "40x40@0.0500+0.000,0.000") is False


def test_the_cloud_id_is_read_separately_from_the_map_id():
    """The bug this pair of functions exists to prevent."""
    header = an_agreement()
    assert bridge.cloud_map_id(header) == 3
    assert header["map_id"] != 3
    # A cloud reset must not read as a robot-map change, or the robot drops the
    # wrong state on the wrong event.
    header["cloud_map_id"] = 4
    assert bridge.is_stale(header, header["map_id"]) is False


def test_a_missing_cloud_id_is_none_not_zero():
    header = an_agreement()
    header.pop("cloud_map_id")
    assert bridge.cloud_map_id(header) is None


# --- routing `found` -------------------------------------------------------

def test_a_live_sighting_is_perception():
    assert bridge.found_topic(a_found(basis="live")) == "detections"


def test_a_remembered_sighting_is_a_memory():
    assert bridge.found_topic(a_found(basis="memory")) == "target"


def test_absent_goes_to_neither_topic():
    """An empty array reads as 'nothing this frame', which is a blank wall too."""
    assert bridge.found_topic(a_found(found=False, basis="absent")) is None


def test_found_false_goes_nowhere_whatever_the_basis_says():
    assert bridge.found_topic(a_found(found=False, basis="live")) is None


def test_the_height_survives_into_the_hypothesis():
    """It is the number the 2D map cannot supply and the whole point of the cloud."""
    hyp = bridge.found_hypothesis(a_found(z=0.74))
    assert hyp["z"] == 0.74


def test_reachability_stays_three_valued():
    """`None` means go and look; `False` means do not bother.  Not the same."""
    assert bridge.found_hypothesis(a_found(reachable=None))["reachable"] is None
    assert bridge.found_hypothesis(a_found(reachable=False))["reachable"] is False
    assert bridge.found_hypothesis(a_found(reachable=True))["reachable"] is True


def test_the_approach_pose_defaults_to_the_target_when_absent():
    header = a_found()
    header.pop("approach")
    hyp = bridge.found_hypothesis(header)
    assert hyp["approach_x"] == hyp["x"]
    assert hyp["approach_y"] == hyp["y"]


# --- the pose hint ---------------------------------------------------------

def test_a_weak_hint_is_not_worth_reporting():
    header = {"advisory": True, "confidence": 0.1, "inliers": 5}
    assert bridge.pose_hint_is_usable(header, 0.5, 40) is False


def test_a_strong_hint_is():
    header = {"advisory": True, "confidence": 0.9, "inliers": 120}
    assert bridge.pose_hint_is_usable(header, 0.5, 40) is True


def test_a_non_advisory_hint_is_refused_outright():
    """SLAM owns the robot's pose; a server claiming otherwise is not understood."""
    with pytest.raises(bridge.BridgeError, match="advisory"):
        bridge.pose_hint_is_usable({"advisory": False, "confidence": 1.0,
                                    "inliers": 500})


# --- the deployed-client version guard -------------------------------------

class TestTheClientVersionGuard:
    """
    A client older than this node must be refused with a usable message.

    This pair is the one place where two repositories must move together and
    only one of them is a git dependency: the node arrives by `git pull`, the
    client file by `scp` from RoboCamStreamProcessing/link. Updating one and
    not the other is the normal mistake, and unguarded it surfaces as a
    TypeError from inside a constructor -- naming the symptom, not the cause.
    """

    def a_module(self, **kwargs):
        import types
        mod = types.ModuleType("fake_robocam_client")

        class RoboCamClient:
            def __init__(self, server, client_id="orin", **rest):
                pass

        # Rebuild __init__ with the given keyword names so inspect sees them.
        names = ", ".join(f"{k}=None" for k in kwargs)
        src = f"def __init__(self, server, client_id='orin'{',' if names else ''} {names}):\n    pass\n"
        ns = {}
        exec(src, ns)
        RoboCamClient.__init__ = ns["__init__"]
        mod.RoboCamClient = RoboCamClient
        return mod

    def test_a_current_client_passes(self):
        mod = self.a_module(**{k: None for k in bridge.REQUIRED_CLIENT_KWARGS})
        bridge.check_client_api(mod, "/tmp/robocam_client.py")   # must not raise

    def test_an_old_client_is_refused_by_name(self):
        mod = self.a_module()          # a v1 client: none of the kwargs
        with pytest.raises(RuntimeError) as exc:
            bridge.check_client_api(mod, "/tmp/robocam_client.py")
        message = str(exc.value)
        assert "older than this node" in message
        assert "map_every_s" in message          # says which is missing
        assert "scp" in message                  # says how to fix it
        assert "enable_map_loop:=false" in message   # and how to proceed now

    def test_a_partially_updated_client_is_also_refused(self):
        """Half the map loop is not a working map loop."""
        mod = self.a_module(map_every_s=None, on_map_update=None)
        with pytest.raises(RuntimeError, match="on_agreement"):
            bridge.check_client_api(mod, "/tmp/robocam_client.py")

    def test_a_client_that_can_carry_a_frame_pose_is_recognised(self):
        mod = self.a_module()

        def _send_frame(self, img, pre_encoded, cv2, pose=None):
            pass

        mod.RoboCamClient._send_frame = _send_frame
        assert bridge.client_takes_frame_pose(mod) is True

    def test_an_older_client_is_not_asked_to_carry_one(self):
        """Not refused: it still runs the loop, and loses only the frame pose."""
        mod = self.a_module()

        def _send_frame(self, img, pre_encoded, cv2):
            pass

        mod.RoboCamClient._send_frame = _send_frame
        assert bridge.client_takes_frame_pose(mod) is False
        assert bridge.client_takes_frame_pose(object()) is False
