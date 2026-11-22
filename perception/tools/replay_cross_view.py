"""Replay recorded capture-contract topics into a fixed, isolated ROS namespace."""
import argparse
import json
import time
import uuid

from utils.frame_source import CAPTURE_PREFIX, capture_identity


def rebase_source(frame_id, stamp_ns, offset_ns, run):
    identity = capture_identity(frame_id)
    if not identity or not identity["frame_valid"]:
        return frame_id, stamp_ns
    fields = frame_id[len(CAPTURE_PREFIX):].split("|")
    fields[2] = uuid.uuid5(run, fields[2]).hex
    return CAPTURE_PREFIX + "|".join(fields), stamp_ns + offset_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="Directory of a rosbag2 recording with SQLite3 storage")
    parser.add_argument("rig", help="JSON file containing the original rig topics")
    args = parser.parse_args()

    import rclpy
    import rosbag2_py
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    with open(args.rig, encoding="utf-8") as handle:
        rig = json.load(handle)
    routes = {}
    for side in "ab":
        for role, topic in rig["topics"][side].items():
            if role not in ("rgb", "objects", "depth") or topic in routes:
                raise ValueError("invalid or duplicate recorded topic")
            routes[topic] = (f"/cross_view_replay/{side}/{role}",
                             String if role == "objects" else CompressedImage)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    recorded = {item.name: item.type for item in reader.get_all_topics_and_types()}
    for topic, (_, kind) in routes.items():
        expected = "std_msgs/msg/String" if kind is String else "sensor_msgs/msg/CompressedImage"
        if recorded.get(topic) != expected:
            raise ValueError(f"{topic} must have recorded type {expected}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(routes)))

    rclpy.init()
    node = Node("cross_view_recording_replay")
    try:
        publishers = {topic: node.create_publisher(kind, dest, qos_profile_sensor_data)
                      for topic, (dest, kind) in routes.items()}
        run = uuid.uuid4()
        first = None
        time.sleep(2)  # Allow discovery after publishers are created.
        while reader.has_next() and rclpy.ok():
            topic, encoded, recorded_ns = reader.read_next()
            if first is None:
                first = recorded_ns
                offset = time.time_ns() - first
                began = time.monotonic_ns()
            delay = (began + recorded_ns - first - time.monotonic_ns()) / 1e9
            if delay > 0:
                time.sleep(delay)
            kind = routes[topic][1]
            message = deserialize_message(encoded, kind)
            if kind is String:
                payload = json.loads(message.data)
                source = payload.get("source")
                if not isinstance(source, dict) or not isinstance(source.get("stamp_ns"), int):
                    continue
                source["frame_id"], source["stamp_ns"] = rebase_source(
                    source.get("frame_id"), source["stamp_ns"], offset, run)
                identity = capture_identity(source["frame_id"])
                if identity is not None:
                    source.update(identity)
                message.data = json.dumps(payload)
            else:
                header = message.header
                stamp = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
                header.frame_id, stamp = rebase_source(header.frame_id, stamp, offset, run)
                header.stamp.sec, header.stamp.nanosec = divmod(stamp, 1_000_000_000)
            publishers[topic].publish(message)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
