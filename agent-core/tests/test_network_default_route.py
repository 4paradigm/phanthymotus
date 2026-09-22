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
