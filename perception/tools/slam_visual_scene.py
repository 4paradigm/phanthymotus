#!/usr/bin/env python3
"""Deterministic metric RGB/scan scene for native loops and pose-loss replay.

No robot APIs. Images/depth are rendered from a fixed textured, non-planar wall.
The manifest records truth separately from drifted/reset external odometry.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def generate(root, origin=(0., 0., 0.), *, resume=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    (root/'origin.txt').write_text(' '.join(map(str, origin))+'\n')
    rng = np.random.default_rng(371)
    texture = rng.integers(0, 256, (1200, 2400, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), .8)
    # Stable corners at multiple scales, anchored to the world's surface.
    for _ in range(3500):
        x, y = rng.integers(10, 2390), rng.integers(10, 1190)
        cv2.circle(texture, (int(x), int(y)), int(rng.integers(2, 10)),
                   tuple(int(v) for v in rng.integers(0, 256, 3)), -1)
    height, width, f = 240, 320, 260.
    v, u = np.indices((height, width), dtype=np.float32)
    rx, ry = (u-width/2)/f, (v-height/2)/f
    route = []
    for i, x in enumerate(np.arange(0, 5.5, .18)):
        route.append((float(x), True, float(x+i*.012), 'outbound'))
    for i, x in enumerate(np.arange(5.4, -.01, -.18)):
        route.append((float(x), True, float(x+(i+31)*.012), 'revisit'))
    # Appearance-identical but metrically incompatible observation, then novel view.
    route += [(0., False, 0., 'similar_negative')]*4
    route += [(15., False, 0., 'novel_negative')]*4
    route += [(x, False, 0., 'relocalize') for x in (0., .01, .02, .03)]
    route += [(float(x), False, 0., 'visual_only') for x in np.arange(.2, 6.7, .18)]
    route += [(float(x), True, float(100+x), 'external_reset') for x in np.arange(6.8, 7.6, .18)]
    route += [(7.6, True, 107.6, 'external_rgb_gap')]
    route += [(float(x), True, float(-100+x), 'continuous_reset') for x in np.arange(7.7, 8.3, .18)]
    route += [(7.5, False, 0., 'rgb_gap')]*3
    route += [(x, False, 0., 'second_relocalize') for x in (0., .01, .02, .03)]
    route += [(float(x), False, 0., 'second_visual_only') for x in np.arange(.2, 1.5, .18)]
    if resume:
        route = [(i*.18, i>=10, 100+i*.18, 'depth_gap' if i in (4,5) else 'reopened') for i in range(16)]
    with (root/'frames.txt').open('w') as manifest:
        for i, (x, external, odom, phase) in enumerate(route):
            z = np.full_like(rx, 3.)
            for _ in range(16):
                wx, wy = x+rx*z, ry*z
                z = 3.+.24*np.sin(wx*1.7)+.18*np.cos(wy*2.3)+.14*np.sin(wx*3+wy)
            wx, wy = x+rx*z, ry*z
            rgb = cv2.remap(texture, (wx*100+350).astype('float32'),
                            (wy*100+600).astype('float32'), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            if phase == 'similar_negative':
                # Same rendered texture as origin, but no rigid depth agreement.
                z = 12.+2.*np.sin(u*.17)*np.cos(v*.13)
            cv2.imwrite(str(root/f'{i}.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(root/f'{i}.png'), np.rint(z*1000).astype('uint16'))
            manifest.write(f'{i} {int(external)} {odom:.6f} {x:.6f} {phase}\n')
    return len(route)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--origin', nargs=3, type=float, default=(0.,0.,0.))
    parser.add_argument('--resume', action='store_true', help='16-frame saved-map recovery scene')
    args=parser.parse_args()
    print(generate(args.output,args.origin,resume=args.resume))
