"""Optional non-visual JPEG frame provenance for transport measurements.

APP15 is ignored by ordinary JPEG decoders. Source timestamps retain their
original clock; callers must establish a timebase before computing latency.
"""
import json
import struct
import time

SIGNATURE = b"FEL1\x00"


def tag_jpeg(jpeg, metadata):
    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("expected JPEG")
    payload = {"schema":"motus.face_effects/frame/1", "frame_id":metadata["frame_id"],
               "source_timestamp":metadata["source_timestamp"],
               "source_frame_id":str(metadata.get("source_metadata",{}).get("ros_frame_id",""))[:512],
               "encoded_unix_ns":time.time_ns()}
    data = SIGNATURE + json.dumps(payload,separators=(",",":"),allow_nan=False).encode()
    if len(data)>4096:
        raise ValueError("JPEG telemetry too large")
    return jpeg[:2]+b"\xff\xef"+struct.pack(">H",len(data)+2)+data+jpeg[2:]


def read_tag(jpeg):
    if not jpeg.startswith(b"\xff\xd8"):
        return None
    pos=2
    while pos+4<=len(jpeg) and jpeg[pos]==255:
        marker=jpeg[pos+1]
        if marker in (0xda,0xd9):
            return None
        length=struct.unpack(">H",jpeg[pos+2:pos+4])[0]
        if length<2 or pos+2+length>len(jpeg):
            return None
        payload=jpeg[pos+4:pos+2+length]
        if marker==0xef and payload.startswith(SIGNATURE):
            try:
                return json.loads(payload[len(SIGNATURE):])
            except (ValueError,UnicodeDecodeError):
                return None
        pos+=2+length
    return None
