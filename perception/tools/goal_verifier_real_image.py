"""Run the actual CPU detector on a recorded photograph, without retiming it.

An image file alone does not prove a camera identity or capture timestamp.
Output deliberately includes an inadmissible observation for negative protocol
testing; detected pixels/boxes are real measurements, not hand-written goals.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from plugins.goal_verifier.sources import color_objects
from plugins.goal_verifier.contracts import OBSERVATION


def inspect_image(image, classes):
    path = Path(image)
    pixels = cv2.imread(str(path))
    if pixels is None:
        raise ValueError("IMAGE_DECODE_FAILED")
    objects = color_objects(pixels, classes)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(
        scope="REAL_PHOTOGRAPH_ACTUAL_CPU_COLOR_DETECTOR",
        image_sha256=sha,
        image_size=[pixels.shape[1], pixels.shape[0]],
        objects=objects,
        detector="configured HSV segmentation, not official VOP or semantic recognition",
        capture_timestamp_known=False,
        camera_identity_known=False,
        trusted_goal_verification="BLOCKED: original capture/camera evidence absent",
        inadmissible_observation=dict(
            schema=OBSERVATION,
            observation_id="file-" + sha,
            image_size=[pixels.shape[1], pixels.shape[0]],
            objects=objects,
            provenance="replay",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--classes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = inspect_image(a.image, json.loads(a.classes.read_text()))
    with a.output.open("x") as out:
        json.dump(result, out, indent=2)
    print(
        json.dumps({k: v for k, v in result.items() if k != "inadmissible_observation"})
    )
