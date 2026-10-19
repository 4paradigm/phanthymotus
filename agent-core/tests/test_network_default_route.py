"""_default_routes / _uplink_warning — telling association apart from connectivity."""
import os, sys, textwrap, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from api import network as N


def _route_file(tmp_path, body):
    p = tmp_path / "route"
    p.write_text("Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\n" + body)
    return str(p)


def _patch(monkeypatch, path):
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda f, *a, **k: real_open(path if f == "/proc/net/route" else f, *a, **k))


R1SZ = ("eth10\t00000000\t017BA8C0\t0003\t0\t0\t20100\t00000000\n"
        "wlan0\t00000000\t0180640A\t0003\t0\t0\t20600\t00000000\n"
        "eth10\t0000A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\n")


def test_only_default_routes_are_collected(tmp_path, monkeypatch):
    _patch(monkeypatch, _route_file(tmp_path, R1SZ))
    assert N._default_routes() == {"eth10": 20100, "wlan0": 20600}


def test_the_lowest_metric_wins(tmp_path, monkeypatch):
    """r1_sz exactly: the body link beats the office WiFi, so nothing reaches
    the internet while both devices report a gateway."""
    _patch(monkeypatch, _route_file(tmp_path, R1SZ))
    assert N._internet_device() == "eth10"


def test_no_routes_at_all(tmp_path, monkeypatch):
    _patch(monkeypatch, _route_file(tmp_path, ""))
    assert N._default_routes() == {} and N._internet_device() == ""


def test_a_missing_proc_file_is_not_fatal(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert N._default_routes() == {}


def test_wifi_winning_produces_no_warning(tmp_path, monkeypatch):
    """The ordinary case must read exactly as it did before."""
    _patch(monkeypatch, _route_file(
        tmp_path, "wlan0\t00000000\t0180640A\t0003\t0\t0\t600\t00000000\n"))
    assert N._uplink_warning("wlan0", settle_s=0) == ""


def test_a_lower_metric_ethernet_is_named(tmp_path, monkeypatch):
    _patch(monkeypatch, _route_file(tmp_path, R1SZ))
    msg = N._uplink_warning("wlan0", settle_s=0)
    assert "eth10" in msg and "20100" in msg and "本体" in msg


def test_wifi_with_no_default_route_is_reported(tmp_path, monkeypatch):
    _patch(monkeypatch, _route_file(
        tmp_path, "eth10\t00000000\t017BA8C0\t0003\t0\t0\t20100\t00000000\n"))
    assert "没有拿到默认路由" in N._uplink_warning("wlan0", settle_s=0)


def test_no_default_route_anywhere_is_reported(tmp_path, monkeypatch):
    _patch(monkeypatch, _route_file(tmp_path, ""))
    assert "没有任何默认路由" in N._uplink_warning("wlan0", settle_s=0)


# ── the choice has to survive a reboot ───────────────────────────────────────
#
# Setting the metric is only half of it. The metric itself lives in the
# NetworkManager profile and does persist — but netplan regenerates the *other*
# interface's static default route at boot, and on r1_sz that route came back
# with the lower metric and took the uplink with it. The machine was then
# offline again, showing up as every LLM call timing out after 5s while WiFi
# reported connected and healthy.


class _FakeConfig(dict):
    pass


def test_nothing_happens_when_no_preference_was_ever_set(monkeypatch):
    """A machine that never had this problem must be left alone."""
    monkeypatch.setattr(N.config, "main", _FakeConfig())
    assert N.ensure_preferred_uplink() == {"checked": False}


def test_a_satisfied_preference_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(N.config, "main", _FakeConfig({"preferred_uplink": "wlan0"}))
    monkeypatch.setattr(N, "_internet_device", lambda: "wlan0")
    called = []
    monkeypatch.setattr(N, "_set_route_priority_sync",
                        lambda *a: called.append(a))
    out = N.ensure_preferred_uplink()
    assert out["changed"] is False and called == []


def test_a_lost_preference_is_re_applied(monkeypatch):
    """The reboot case: netplan put the dead route back and it won again."""
    monkeypatch.setattr(N.config, "main", _FakeConfig({"preferred_uplink": "wlan0"}))
    seen = iter(["eth10", "wlan0"])       # before the fix, then after
    monkeypatch.setattr(N, "_internet_device", lambda: next(seen))
    called = []
    monkeypatch.setattr(N, "_set_route_priority_sync",
                        lambda *a: called.append(a))
    out = N.ensure_preferred_uplink()
    assert called == [("wlan0", True)]
    assert out["changed"] is True and out["ok"] is True


def test_a_failure_is_reported_not_raised(monkeypatch):
    """This runs during startup. A robot that will not boot because its network
    could not be tidied up is a far worse outcome than a wrong default route."""
    monkeypatch.setattr(N.config, "main", _FakeConfig({"preferred_uplink": "wlan0"}))
    monkeypatch.setattr(N, "_internet_device", lambda: "eth10")

    def _boom(*_a):
        raise RuntimeError("NetworkManager is not running")

    monkeypatch.setattr(N, "_set_route_priority_sync", _boom)
    assert "NetworkManager" in N.ensure_preferred_uplink()["error"]
