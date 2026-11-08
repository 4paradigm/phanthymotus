"""Chessboard intrinsic calibration for the S10 Pro RTSP cameras.

Usage:
    python calibrate_chessboard.py <frame_dir> --out <out_dir>
        [--cols 10 --rows 13 --square 0.018]
        [--camera-id deeprobotics/s10_pro/camera_front] [--camera-desc "..."]
        [--target-width 640 --target-height 360]

Detects inner corners on every frame, drops blurry / undetected / outlier
frames, calibrates with plumb_bob (5 coeffs), rational_polynomial (8) and
equidistant (fisheye, 4), reports RMS and per-frame error, and writes the
undistort-card `calibration` JSON at capture resolution and scaled to the
driver output resolution, plus `table_entry.json`: one row for
`perception/plugins/undistortion_runtime.CALIBRATIONS`, ready to paste.
"""
import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(2)


def find_corners(gray, pattern):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    try:
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if ok:
            return corners
    except cv2.error:
        pass
    ok, corners = cv2.findChessboardCorners(
        gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def sharpness(gray, corners):
    x0, y0 = corners.reshape(-1, 2).min(0).astype(int)
    x1, y1 = corners.reshape(-1, 2).max(0).astype(int)
    roi = gray[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def per_view_errors(obj, img, k, d, rvecs, tvecs, fisheye):
    errs = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        if fisheye:
            proj, _ = cv2.fisheye.projectPoints(o.reshape(1, -1, 3).astype(np.float64), r, t, k, d)
        else:
            proj, _ = cv2.projectPoints(o, r, t, k, d)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1)))))
    return errs


def calibrate(model, obj, img, size):
    if model == "equidistant":
        k = np.eye(3)
        d = np.zeros((4, 1))
        fx = cv2.fisheye if hasattr(cv2.fisheye, "CALIB_FIX_SKEW") else cv2  # OpenCV 4 vs 5
        flags = fx.CALIB_RECOMPUTE_EXTRINSIC | fx.CALIB_FIX_SKEW | fx.CALIB_CHECK_COND
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            [o.reshape(1, -1, 3).astype(np.float64) for o in obj],
            [i.reshape(1, -1, 2).astype(np.float64) for i in img],
            size, k, d, flags=flags, criteria=crit)
        return rms, k, d.reshape(-1), rvecs, tvecs, True
    flags = cv2.CALIB_FIX_K3 if model == "plumb_bob4" else 0
    if model == "rational_polynomial":
        flags = cv2.CALIB_RATIONAL_MODEL
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    d = d.reshape(-1)
    if model == "plumb_bob":
        d = d[:5]
    elif model == "rational_polynomial":
        d = d[:8]
    return rms, k, d, rvecs, tvecs, False


def to_card(camera_id, model, k, d, width, height):
    return {
        "id": camera_id,
        "width": int(width),
        "height": int(height),
        "distortion_model": model,
        "K": [float(k[0, 0]), 0, float(k[0, 2]), 0, float(k[1, 1]), float(k[1, 2]), 0, 0, 1],
        "D": [float(v) for v in d],
    }


def scale_card(card, tw, th):
    sx = tw / card["width"]
    sy = th / card["height"]
    if abs(sx - sy) > 1e-6:
        raise SystemExit(f"target {tw}x{th} is not an isotropic scale of {card['width']}x{card['height']}")
    k = card["K"]
    out = dict(card)
    out["width"], out["height"] = tw, th
    out["K"] = [k[0] * sx, 0, (k[2] + 0.5) * sx - 0.5, 0, k[4] * sy, (k[5] + 0.5) * sy - 0.5, 0, 0, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=10, help="inner corners along the short edge")
    ap.add_argument("--rows", type=int, default=13, help="inner corners along the long edge")
    ap.add_argument("--square", type=float, default=0.018, help="square size in metres")
    ap.add_argument("--camera-id", default="deeprobotics/s10_pro/camera_front")
    ap.add_argument("--camera-desc", default="",
                    help="human description of the lens for the table row's `camera` field")
    ap.add_argument("--measured-on", default="",
                    help="unit and date for the table row (default: today's date)")
    ap.add_argument("--target-width", type=int, default=640)
    ap.add_argument("--target-height", type=int, default=360)
    ap.add_argument("--blur-keep", type=float, default=0.35,
                    help="drop frames whose sharpness is below this fraction of the median")
    ap.add_argument("--outlier", type=float, default=2.0,
                    help="drop frames whose reprojection error exceeds this multiple of the median")
    ap.add_argument("--max-frames", type=int, default=80)
    args = ap.parse_args()

    frame_dir = Path(args.frame_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"no frames in {frame_dir}")

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    size = None
    detected = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"skip unreadable {p.name}")
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
            print(f"frame size {w}x{h}")
        elif (w, h) != size:
            raise SystemExit(f"{p.name} is {w}x{h}, expected {size[0]}x{size[1]}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, pattern)
        if corners is None:
            continue
        detected.append((p, corners.astype(np.float32), sharpness(gray, corners)))
    print(f"detected board in {len(detected)}/{len(files)} frames")
    if len(detected) < 10:
        raise SystemExit("too few detections; capture more frames")

    med_sharp = float(np.median([s for _, _, s in detected]))
    kept = [(p, c, s) for p, c, s in detected if s >= args.blur_keep * med_sharp]
    print(f"sharpness median {med_sharp:.1f}; kept {len(kept)} after blur filter")

    # Thin near-duplicate poses: keep frames whose board centre moved or whose
    # extent changed noticeably versus the previously kept frame.
    thinned = []
    last = None
    for p, c, s in kept:
        pts = c.reshape(-1, 2)
        centre = pts.mean(0)
        extent = pts.max(0) - pts.min(0)
        key = np.concatenate([centre, extent])
        if last is None or np.linalg.norm(key - last) > 12.0:
            thinned.append((p, c, s))
            last = key
    kept = thinned
    print(f"kept {len(kept)} after near-duplicate thinning")
    if len(kept) > args.max_frames:
        step = len(kept) / args.max_frames
        kept = [kept[int(i * step)] for i in range(args.max_frames)]
        print(f"subsampled to {len(kept)} frames")

    # Coverage report: 4x3 grid of board centres and corner heatmap.
    grid = np.zeros((3, 4), int)
    heat = np.zeros((6, 8), int)
    for _, c, _ in kept:
        pts = c.reshape(-1, 2)
        cx, cy = pts.mean(0)
        grid[min(int(cy / size[1] * 3), 2), min(int(cx / size[0] * 4), 3)] += 1
        for x, y in pts:
            heat[min(int(y / size[1] * 6), 5), min(int(x / size[0] * 8), 7)] += 1
    print("board-centre coverage (rows top->bottom, cols left->right):")
    print(grid)
    print("corner coverage heatmap (6x8 cells, 0 = no corners seen there):")
    print(heat)
    empty = int((heat == 0).sum())
    if empty:
        print(f"WARNING: {empty} heatmap cells never saw a corner; edges/corners under-covered")

    obj = [objp.copy() for _ in kept]
    img = [c for _, c, _ in kept]
    names = [p.name for p, _, _ in kept]

    results = {}
    for model in ("plumb_bob", "rational_polynomial", "equidistant"):
        try:
            rms, k, d, rvecs, tvecs, fe = calibrate(model, obj, img, size)
        except cv2.error as exc:
            print(f"{model}: failed ({str(exc).splitlines()[0]})")
            continue
        errs = per_view_errors(obj, img, k, d, rvecs, tvecs, fe)
        med = float(np.median(errs))
        keep_idx = [i for i, e in enumerate(errs) if e <= args.outlier * med]
        dropped = [names[i] for i in range(len(errs)) if i not in keep_idx]
        if dropped and len(keep_idx) >= 10:
            rms, k, d, rvecs, tvecs, fe = calibrate(
                model, [obj[i] for i in keep_idx], [img[i] for i in keep_idx], size)
            errs = per_view_errors([obj[i] for i in keep_idx], [img[i] for i in keep_idx],
                                   k, d, rvecs, tvecs, fe)
        results[model] = dict(rms=float(rms), k=k, d=d, errs=errs, dropped=dropped,
                              frames=[names[i] for i in keep_idx])
        print(f"{model:20s} rms={rms:.4f} px  max_view={max(errs):.3f}  "
              f"frames={len(errs)} dropped={len(dropped)}  fx={k[0,0]:.1f} fy={k[1,1]:.1f} "
              f"cx={k[0,2]:.1f} cy={k[1,2]:.1f}  D={np.round(d, 5).tolist()}")

    if not results:
        raise SystemExit("all models failed")

    # Prefer the simplest model unless a richer one is clearly better.
    order = ["plumb_bob", "rational_polynomial", "equidistant"]
    best = min((m for m in order if m in results), key=lambda m: results[m]["rms"])
    if "plumb_bob" in results and results["plumb_bob"]["rms"] <= results[best]["rms"] * 1.15:
        best = "plumb_bob"
    print(f"selected model: {best}")

    report = {"frame_size": size, "pattern": pattern, "square_m": args.square,
              "frames_used": len(kept), "models": {}}
    for m, r in results.items():
        card = to_card(args.camera_id, m, r["k"], r["d"], *size)
        report["models"][m] = dict(rms=r["rms"], per_view=dict(zip(r["frames"], r["errs"])),
                                   dropped=r["dropped"], calibration=card,
                                   calibration_scaled=scale_card(card, args.target_width, args.target_height))
    report["selected"] = best
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sel = report["models"][best]
    (out_dir / "calibration_capture.json").write_text(
        json.dumps(sel["calibration"], indent=2), encoding="utf-8")
    (out_dir / "calibration_card.json").write_text(
        json.dumps(sel["calibration_scaled"], indent=2), encoding="utf-8")

    # One row of perception/plugins/undistortion_runtime.CALIBRATIONS, ready to
    # paste: the scaled calibration plus where it came from. The provenance
    # fields are what make a table row checkable later — which lens, which
    # day, how many frames, how good the fit was.
    entry = dict(sel["calibration_scaled"])
    entry["K"] = [round(v, 6) if isinstance(v, float) else v for v in entry["K"]]
    entry.update({
        "camera": args.camera_desc or args.camera_id,
        "measured_on": args.measured_on or date.today().isoformat(),
        "method": (f"chessboard {args.cols}x{args.rows} inner corners, "
                   f"{args.square * 1000:g} mm squares; perception/tools/calibrate_chessboard.py"),
        "source_resolution": {"width": size[0], "height": size[1]},
        "frames_used": len(results[best]["frames"]),
        "rms_px": round(results[best]["rms"], 3),
        "note": (f"Calibrated on the {size[0]}x{size[1]} stream and scaled to "
                 f"{args.target_width}x{args.target_height} (K x {args.target_width / size[0]:g}, "
                 "D unchanged)."),
    })
    (out_dir / "table_entry.json").write_text(
        json.dumps({args.camera_id: entry}, indent=4, ensure_ascii=False), encoding="utf-8")

    # Visual check: side-by-side original / undistorted for a few frames.
    k = results[best]["k"]
    d = results[best]["d"]
    fe = best == "equidistant"
    if fe:
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, size, cv2.CV_16SC2)
    else:
        m1, m2 = cv2.initUndistortRectifyMap(k, d, None, k, size, cv2.CV_16SC2)
    previews = kept[:: max(1, len(kept) // 6)][:6]
    for p, c, _ in previews:
        im = cv2.imread(str(p))
        und = cv2.remap(im, m1, m2, cv2.INTER_LINEAR)
        cv2.drawChessboardCorners(im, pattern, c, True)
        cv2.imwrite(str(out_dir / f"preview_{p.stem}.jpg"), np.hstack([im, und]))
    hfov = math.degrees(math.atan(k[0, 2] / k[0, 0]) + math.atan((size[0] - k[0, 2]) / k[0, 0]))
    vfov = math.degrees(math.atan(k[1, 2] / k[1, 1]) + math.atan((size[1] - k[1, 2]) / k[1, 1]))
    print(f"pinhole FOV of the {size[0]}x{size[1]} stream: {hfov:.1f} deg x {vfov:.1f} deg")
    print(f"wrote {out_dir / 'calibration_card.json'} ({args.target_width}x{args.target_height}) "
          f"and {len(previews)} previews")


if __name__ == "__main__":
    sys.exit(main())
