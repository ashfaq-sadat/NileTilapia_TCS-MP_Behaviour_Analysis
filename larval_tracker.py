"""
Larval behaviour tracker for 6-well plates (one larva per well).

Processes every video in VIDEO_FOLDER and writes:
  - larva_summary.csv     one row per larva (treatment, replicate, well, endpoints)
  - larva_per_minute.csv  the same endpoints in time bins
  - <video>_tracks.jpg    trajectory image for checking the tracking
  - <video>_wells.jpg     well detection check

Requirements (Windows or Linux):  pip install opencv-python numpy pandas
Run:  python larval_tracker.py
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
VIDEO_FOLDER = "videos"          # folder holding Control.mp4, DMSO.mp4, ...
OUTPUT_FOLDER = "results"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Calibration. Set PX_PER_MM once you have measured it (see notes at the end).
# If None, it is estimated from the detected well size and NOMINAL_WELL_DIAMETER_MM,
# and the 'calibration' column in the output says "estimated".
PX_PER_MM = None
NOMINAL_WELL_DIAMETER_MM = 34.8  # replace with your plate's value when confirmed

ACCLIMATION_S = 2          # seconds excluded from the start (camera shake when recording starts)
BIN_S = 60                 # time-bin length for the per-minute file

ACTIVE_SPEED_MM_S = 2.0    # speed above which a larva counts as active
JITTER_MM = 0.15           # steps smaller than this are treated as no movement
MAX_STEP_MM = 8.0          # larger jumps in one frame are rejected as tracking errors
MIN_BOUT_FRAMES = 2        # an active bout must last at least this many frames
MAX_GAP_FRAMES = 10        # missing detections up to this length are interpolated
CENTRE_RADIUS_FRACTION = 0.707  # centre zone radius / well-floor radius (0.707 = equal areas)

DIFF_THRESHOLD = 25        # grey-level difference from background that counts as larva
MIN_BLOB_AREA_PX = 15
TRACK_RADIUS_FRACTION = 1.02  # tracking area radius / well-floor radius (excludes wall reflections)
BACKGROUND_SAMPLES = 100

# Wells are numbered left to right, top row (W1-W3) then bottom row (W4-W6).
REPLICATE_OF_WELL = {1: "R1", 4: "R1", 2: "R2", 5: "R2", 3: "R3", 6: "R3"}
# --------------------------------------------------------------------------


def build_background(cap, n_frames):
    idx = np.linspace(0, n_frames - 1, min(BACKGROUND_SAMPLES, n_frames)).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(frames, axis=0).astype(np.uint8)


def detect_wells(background):
    """Find the 6 wells as bright regions inside the dark plate, then the floor ring in each.

    Returns a list of dicts ordered W1-W6 (top row left to right, then bottom row) with
    the well's pixel mask (tracking area) and the floor-ring centre and radius (zones, scale).
    """
    blurred = cv2.GaussianBlur(background, (9, 9), 0)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    plate = ((labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(plate)
    cv2.drawContours(hull, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 255, -1)
    bright = cv2.bitwise_and(cv2.bitwise_not(plate), hull)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(bright)
    if n - 1 < 6:
        raise RuntimeError("Could not find 6 wells. Check that the plate is fully in view.")
    biggest = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:6]

    # order as a 2 x 3 grid
    order = sorted(biggest, key=lambda k: cents[k][1])
    order = sorted(order[:3], key=lambda k: cents[k][0]) + sorted(order[3:], key=lambda k: cents[k][0])

    grey = cv2.medianBlur(background, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def floor_ring(k, rmin, rmax):
        cx, cy = cents[k]
        R = np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi)
        pad = int(R + 10)
        x0, y0 = max(int(cx) - pad, 0), max(int(cy) - pad, 0)
        sub = grey[y0:int(cy) + pad, x0:int(cx) + pad]
        c = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10 * R, param1=80,
                             param2=30, minRadius=int(rmin), maxRadius=int(rmax))
        if c is None:
            return float(cx), float(cy), 0.84 * R           # fallback: centre of bright region
        x, y, r = c[0][0]
        return float(x + x0), float(y + y0), float(r)

    radii = [np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi) for k in order]
    first = [floor_ring(k, 0.75 * R, 0.90 * R) for k, R in zip(order, radii)]
    r_med = np.median([f[2] for f in first])
    # second pass with a tight radius range, so every well gets a consistent ring
    rings = [floor_ring(k, 0.95 * r_med, 1.05 * r_med) for k in order]

    wells = []
    for k, (x, y, r) in zip(order, rings):
        mask = cv2.dilate(((labels == k) * 255).astype(np.uint8), kernel)
        # keep only the well floor: blobs further out are mostly reflections in the wall
        floor = np.zeros_like(mask)
        cv2.circle(floor, (int(x), int(y)), int(TRACK_RADIUS_FRACTION * r), 255, -1)
        mask = cv2.bitwise_and(mask, floor)
        wells.append({"mask": mask, "x": x, "y": y, "r": r})
    return wells


def track_video(path):
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    background = build_background(cap, n_frames)
    wells = detect_wells(background)
    ring_r = float(np.median([w["r"] for w in wells]))

    if PX_PER_MM:
        px_per_mm, calibration = PX_PER_MM, "measured"
    else:
        px_per_mm, calibration = 2 * ring_r / NOMINAL_WELL_DIAMETER_MM, "estimated"
    max_step_px = MAX_STEP_MM * px_per_mm

    boxes = []
    for w in wells:
        ys, xs = np.nonzero(w["mask"])
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    bg = background.astype(np.int16)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    positions = np.full((n_frames, 6, 2), np.nan)
    last = [None] * 6
    missed = [0] * 6

    for t in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            positions = positions[:t]
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.clip(bg - grey, 0, 255).astype(np.uint8)  # larvae are darker than background
        _, binary = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        for i, (w, (x0, y0, x1, y1)) in enumerate(zip(wells, boxes)):
            sub = cv2.bitwise_and(binary[y0:y1, x0:x1], w["mask"][y0:y1, x0:x1])
            n, _, stats, cents = cv2.connectedComponentsWithStats(sub)
            cand = [(cents[j] + [x0, y0], stats[j, cv2.CC_STAT_AREA])
                    for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA_PX]
            if not cand:
                missed[i] += 1
                continue
            if last[i] is None or missed[i] > MAX_GAP_FRAMES:
                pt = max(cand, key=lambda c: c[1])[0]   # (re)acquire: take the largest blob
            else:
                pt = min(cand, key=lambda c: np.linalg.norm(c[0] - last[i]))[0]
                if np.linalg.norm(pt - last[i]) > max_step_px * (1 + missed[i]):
                    missed[i] += 1
                    continue                             # reject an impossible jump
            positions[t, i] = pt
            last[i] = pt
            missed[i] = 0
    cap.release()
    return positions, wells, fps, px_per_mm, calibration, background


def endpoints(xy_mm, centre_mm, floor_r_mm, fps):
    steps = np.linalg.norm(np.diff(xy_mm, axis=0), axis=1)
    steps[steps < JITTER_MM] = 0.0
    speed = steps * fps
    valid = ~np.isnan(speed)
    if valid.sum() == 0:
        return None
    smooth = pd.Series(speed).rolling(3, center=True, min_periods=1).median().to_numpy()
    active = (speed > ACTIVE_SPEED_MM_S) & valid

    # movement bouts: runs of active frames lasting at least MIN_BOUT_FRAMES
    bouts, run = 0, 0
    for a in np.append(active, False):
        if a:
            run += 1
        else:
            bouts += run >= MIN_BOUT_FRAMES
            run = 0

    r = np.linalg.norm(xy_mm - centre_mm, axis=1)
    r_valid = r[~np.isnan(r)]
    duration = valid.sum() / fps
    distance = np.nansum(steps)
    return {
        "total_distance_mm": round(distance, 2),
        "mean_speed_mm_s": round(distance / duration, 3),
        "mean_speed_when_active_mm_s": round(np.nanmean(speed[active]), 3) if active.any() else 0.0,
        "max_speed_mm_s": round(np.nanmax(smooth), 2),
        "pct_time_active": round(100 * active.sum() / valid.sum(), 2),
        "n_bouts": int(bouts),
        "pct_time_outer_zone": round(100 * np.mean(r_valid > CENTRE_RADIUS_FRACTION * floor_r_mm), 2),
        "analysed_s": round(duration, 1),
    }


def draw_check_images(stem, background, wells, positions):
    img = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    for i, w in enumerate(wells):
        contours, _ = cv2.findContours(w["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)                  # tracking area
        c = (int(w["x"]), int(w["y"]))
        cv2.circle(img, c, int(w["r"]), (255, 120, 0), 1)                    # floor ring
        cv2.circle(img, c, int(w["r"] * CENTRE_RADIUS_FRACTION), (255, 120, 0), 1)  # centre zone
        cv2.putText(img, f"W{i + 1} {REPLICATE_OF_WELL[i + 1]}", (c[0] - 45, c[1] - int(w["r"]) - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{stem}_wells.jpg"), img)

    colours = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200), (0, 200, 200)]
    for i in range(6):
        pts = positions[:, i]
        pts = pts[~np.isnan(pts[:, 0])].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colours[i], 1)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{stem}_tracks.jpg"), img)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    videos = sorted(p for p in glob.glob(os.path.join(VIDEO_FOLDER, "*"))
                    if p.lower().endswith(VIDEO_EXTENSIONS))
    if not videos:
        print(f"No videos found in '{VIDEO_FOLDER}'.")
        return

    summary, per_bin = [], []
    for path in videos:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"Tracking {stem} ...")
        positions, wells, fps, px_per_mm, calibration, bg = track_video(path)
        draw_check_images(stem, bg, wells, positions)

        start = int(ACCLIMATION_S * fps)
        bin_len = int(BIN_S * fps)

        for w in range(6):
            raw = pd.DataFrame(positions[start:, w])
            detection = 100 * raw[0].notna().mean()
            xy_mm = raw.interpolate(limit=MAX_GAP_FRAMES, limit_area="inside").to_numpy() / px_per_mm
            centre_mm = np.array([wells[w]["x"], wells[w]["y"]]) / px_per_mm
            floor_r_mm = wells[w]["r"] / px_per_mm
            info = {"video": stem, "treatment": stem, "replicate": REPLICATE_OF_WELL[w + 1],
                    "well": f"W{w + 1}"}

            res = endpoints(xy_mm, centre_mm, floor_r_mm, fps)
            if res:
                summary.append({**info, **res, "detection_pct": round(detection, 1),
                                "px_per_mm": round(px_per_mm, 3), "calibration": calibration})

            for b, s in enumerate(range(0, len(xy_mm) - 1, bin_len)):
                seg = xy_mm[s:s + bin_len + 1]
                if len(seg) - 1 < bin_len / 2:
                    continue                          # skip a short final bin
                r = endpoints(seg, centre_mm, floor_r_mm, fps)
                if r:
                    per_bin.append({**info, "bin": b + 1,
                                    "bin_start_s": round(ACCLIMATION_S + s / fps, 1), **r})

    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_FOLDER, "larva_summary.csv"), index=False)
    pd.DataFrame(per_bin).to_csv(os.path.join(OUTPUT_FOLDER, "larva_per_minute.csv"), index=False)
    print(f"Done. Results are in '{OUTPUT_FOLDER}'.")


if __name__ == "__main__":
    main()

# Measuring PX_PER_MM: open one *_wells.jpg (or a video frame) in ImageJ/Fiji, draw a line
# across a well along a known dimension (e.g. the inner well diameter at the bottom),
# read the length in pixels, and divide by that dimension in mm.
