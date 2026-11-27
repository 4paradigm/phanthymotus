"""Recording summaries and read-only file selection without a ROS runtime."""
import os
import sys
from unittest.mock import Mock

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plugins.rosbag import plugin as plugin_module  # noqa: E402
from plugins.rosbag.store import BagStore  # noqa: E402


BAG = "bag_20261127_120000_a1b2c3d4e5f6"


def make_bag(root, count=3):
    directory = root / BAG
    directory.mkdir()
    data_name = BAG + "_0.db3"
    (directory / data_name).write_bytes(b"recorded data")
    metadata = {"rosbag2_bagfile_information": {
        "storage_identifier": "sqlite3", "message_count": count,
        "duration": {"nanoseconds": 2_000_000_000},
        "starting_time": {"nanoseconds_since_epoch": 1_700_000_000_000_000_000},
        "relative_file_paths": [data_name],
        "topics_with_message_count": [
            {"topic_metadata": {"name": "/imu", "type": "sensor_msgs/msg/Imu"},
             "message_count": count}],
    }}
    (directory / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    return directory, data_name


def test_summary_reports_actual_counts_and_missing_requested_topic(tmp_path):
    directory, name = make_bag(tmp_path)
    result = BagStore(tmp_path).inspect(BAG, expected_topics=["/imu", "/camera"])
    assert result["metadata_status"] == "available"
    assert result["message_count"] == 3 and result["duration_sec"] == 2
    assert result["topics"] == [{"name": "/imu", "type": "sensor_msgs/msg/Imu", "message_count": 3}]
    assert result["missing_topics"] == ["/camera"]
    assert {f["name"] for f in result["files"]} == {name, "metadata.yaml"}
    assert result["size_bytes"] == sum(p.stat().st_size for p in directory.iterdir())


def test_empty_bag_and_unknown_original_selection_are_not_reported_as_success(tmp_path):
    make_bag(tmp_path, count=0)
    result = BagStore(tmp_path).inspect(BAG)
    assert result["message_count"] == 0
    assert any("没有记录到消息" in warning for warning in result["warnings"])
    assert any("无法确认是否遗漏" in warning for warning in result["warnings"])
    assert "missing_topics" not in result


def test_empty_bag_with_null_topic_list_is_recognized(tmp_path):
    directory, _ = make_bag(tmp_path, count=0)
    metadata_path = directory / "metadata.yaml"
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    document["rosbag2_bagfile_information"]["topics_with_message_count"] = None
    metadata_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = BagStore(tmp_path).inspect(BAG)
    assert result["metadata_status"] == "available"
    assert result["message_count"] == 0
    assert any("没有记录到消息" in warning for warning in result["warnings"])


def test_original_selection_is_available_after_reopening_store(tmp_path):
    make_bag(tmp_path)
    BagStore(tmp_path).remember_expected(BAG, ["/imu", "/camera"])
    result = BagStore(tmp_path).inspect(BAG)
    assert result["missing_topics"] == ["/camera"]
    assert "rosbag_card.json" in {item["name"] for item in result["files"]}


def test_recordings_can_be_selected_beyond_the_first_page(tmp_path):
    for n in range(3):
        (tmp_path / f"bag_20261127_12000{n}_a1b2c3d4e5f6").mkdir()
    store = BagStore(tmp_path)
    first = store.list_bags(offset=0, limit=2)
    second = store.list_bags(offset=2, limit=2)
    assert first["total"] == 3 and first["has_more"] is True
    assert len(first["recordings"]) == 2 and len(second["recordings"]) == 1
    assert second["has_more"] is False
    with pytest.raises(ValueError):
        store.list_bags(offset=-1)


def test_incomplete_metadata_is_reported_without_fabricating_counts(tmp_path):
    directory, _ = make_bag(tmp_path)
    (directory / "metadata.yaml").write_text("rosbag2_bagfile_information: {}", encoding="utf-8")
    result = BagStore(tmp_path).inspect(BAG, expected_topics=["/imu"])
    assert result["metadata_status"] == "invalid"
    assert result["message_count"] is None
    assert result["warnings"]


def test_invalid_file_list_does_not_expose_partially_parsed_counts(tmp_path):
    directory, _ = make_bag(tmp_path)
    metadata_path = directory / "metadata.yaml"
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    document["rosbag2_bagfile_information"]["relative_file_paths"] = "invalid"
    metadata_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = BagStore(tmp_path).inspect(BAG, expected_topics=["/imu"])
    assert result["metadata_status"] == "invalid"
    assert result["message_count"] is None and result["topics"] == []
    assert "missing_topics" not in result


def test_file_selection_rejects_traversal_and_active_recording(tmp_path):
    _, data_name = make_bag(tmp_path)
    store = BagStore(tmp_path)
    with pytest.raises(ValueError):
        store.open_file("../other", data_name)
    with pytest.raises(FileNotFoundError):
        store.open_file(BAG, "../../secret.db3")
    with pytest.raises(ValueError):
        store.open_file(BAG, data_name, active=True)
    stream, size = store.open_file(BAG, data_name)
    with stream:
        assert stream.read() == b"recorded data"
    assert size == len(b"recorded data")


def test_history_does_not_follow_symlinks(tmp_path):
    outside = tmp_path / "other"
    outside.mkdir()
    link = tmp_path / BAG
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert BagStore(tmp_path).list_bags()["recordings"] == []
    with pytest.raises(FileNotFoundError):
        BagStore(tmp_path).inspect(BAG)


def test_card_info_exposes_history_and_stop_confirms_current_topics(tmp_path, monkeypatch):
    directory, _ = make_bag(tmp_path)
    recorder = Mock()
    snapshot = {"state": "completed", "path": str(directory), "topics": ["/imu", "/camera"]}
    recorder.info.return_value = dict(snapshot)
    recorder.stop.return_value = dict(snapshot)
    monkeypatch.setattr(plugin_module, "Recorder", Mock(return_value=recorder))
    card = plugin_module.RosbagRecorderPlugin({"output_root": str(tmp_path)})
    schema = card.get_tools()[0]["inputSchema"]
    assert schema["x-rosbag-manager"]["version"] == 1
    # Manager query fields do not appear as nonfunctional generic card inputs.
    assert set(schema["properties"]) == {"action"}
    assert card.dispatch(card.PREFIX, {"action": "info", "view": "list"})["recordings"][0]["id"] == BAG
    detail = card.dispatch(card.PREFIX, {"action": "info", "view": "inspect", "bag_id": BAG})
    assert detail["missing_topics"] == ["/camera"]
    stopped = card.dispatch(card.PREFIX, {"action": "stop"})
    assert stopped["result_confirmation"]["missing_topics"] == ["/camera"]


def test_a_failed_but_still_running_process_cannot_be_downloaded(tmp_path, monkeypatch):
    directory, filename = make_bag(tmp_path)
    recorder = Mock()
    recorder.info.return_value = {"state": "error", "process_active": True,
                                  "path": str(directory), "topics": ["/imu"]}
    monkeypatch.setattr(plugin_module, "Recorder", Mock(return_value=recorder))
    card = plugin_module.RosbagRecorderPlugin({"output_root": str(tmp_path)})
    detail = card.dispatch(card.PREFIX, {"action": "info", "view": "inspect", "bag_id": BAG})
    assert detail["recording"] is True
    with pytest.raises(ValueError):
        card.open_recording_file(BAG, filename)


def test_local_file_endpoint_requires_token_and_streams_managed_file(tmp_path, monkeypatch):
    import http.client
    import threading
    import main
    import utils.file_intake

    _, data_name = make_bag(tmp_path)
    card = plugin_module.RosbagRecorderPlugin({"output_root": str(tmp_path)})
    bundle = Mock()
    bundle.open_rosbag_file = card.open_recording_file
    monkeypatch.setattr(main, "_bundle", bundle)
    monkeypatch.setattr(utils.file_intake, "access_token", lambda: "test-token")
    server = main.ThreadingHTTPServer(("127.0.0.1", 0), main.make_handler())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    path = f"/rosbag/file?bag={BAG}&file={data_name}"
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.request("GET", path, headers={"X-Access-Token": "test-token"})
        response = connection.getresponse()
        assert response.status == 200
        assert int(response.getheader("Content-Length")) == len(b"recorded data")
        assert response.read() == b"recorded data"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(2)
