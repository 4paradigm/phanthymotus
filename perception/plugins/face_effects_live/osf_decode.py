"""OSF heatmap/offset decoding, usable inside the isolated ORT host.

Adapted from Emiliana's tracker.py, copyright (c) 2019 Emiliana,
BSD-2-Clause; see licenses/OpenSeeFace.txt and THIRD_PARTY.md.
No image/model state, ROS, or robot dependencies. Coordinates are in the 224px
model input space; mapping them back to the original crop remains in the caller.
"""
import numpy as np


def decode_heatmaps(outputs):
    output = np.asarray(outputs[0])[0]
    if output.shape != (198, 28, 28) or not np.isfinite(output).all():
        raise ValueError('invalid OpenSeeFace output schema/values')
    heat = output[:66].reshape(66, -1)
    arg = heat.argmax(axis=1)[:, None]
    conf = np.take_along_axis(heat, arg, axis=1)[:, 0]
    offsets = []
    for component in (output[66:132], output[132:198]):
        p = np.take_along_axis(component.reshape(66, -1), arg, axis=1)[:, 0]
        p = np.clip(p, 1e-7, 1-1e-7)
        offsets.append(223 * np.log(p/(1-p)) / 16)
    indices = arg[:, 0]
    return (np.column_stack(((indices % 28) * 223/27 + offsets[1],
                             (indices // 28) * 223/27 + offsets[0])), conf)
