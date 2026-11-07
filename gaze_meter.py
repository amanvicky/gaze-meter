#!/usr/bin/env python3
"""gaze-meter — attention analytics from a webcam.

Answers a simple question with numbers instead of vibes: during a session,
how often did I blink, how much of the time was my gaze off the lens, and
how focused was I overall? It is the measurement half of the OpenBroadcast
gaze-correction family (https://github.com/amanvicky/Open-Broadcast): that
project corrects the gaze, this one scores it.

Core metrics, per minute:
  blinks/min       eye-openness dips counted with hysteresis
  off-center %     share of frames with gaze outside the center band
  focus score      100 minus the off-center share — time your eyes spent near the lens

No camera attached, or want a reproducible run? --simulate feeds a
deterministic synthetic session through the exact same math.
"""

import argparse
import csv
import sys
from collections import deque

# 478-point face-mesh indices, same ones openbroadcast-web uses.
L_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
R_EYE = {"outer": 263, "inner": 362, "top": 386, "bottom": 374}
L_IRIS, R_IRIS = 468, 473

OFF_CENTER = 0.28      # |gaze| beyond this (in eye-widths) counts as looking away
CLOSED_BELOW = 0.18    # eye openness under this = eyelid shut
OPEN_ABOVE = 0.28      # reopened past this = blink completed
EMA = 0.35             # gaze smoothing factor, tuned in openbroadcast-web


def eye_box(lm, eye):
    """Center and size of one eye's box from normalized mesh landmarks."""
    outer, inner = lm[eye["outer"]], lm[eye["inner"]]
    top, bottom = lm[eye["top"]], lm[eye["bottom"]]
    return {
        "cx": (outer["x"] + inner["x"]) / 2,
        "cy": (top["y"] + bottom["y"]) / 2,
        "w": max(abs(inner["x"] - outer["x"]), 0.01),
        "h": max(abs(bottom["y"] - top["y"]), 0.01),
    }


def gaze_offset(lm):
    """Iris offset from eye center, normalized by eye width, averaged over both eyes."""
    dxs = dys = 0.0
    for eye, iris in ((L_EYE, L_IRIS), (R_EYE, R_IRIS)):
        box = eye_box(lm, eye)
        iris_pt = lm[iris]
        dxs += (iris_pt["x"] - box["cx"]) / box["w"]
        dys += (iris_pt["y"] - box["cy"]) / box["w"]
    return dxs / 2, dys / 2


def openness(lm):
    """Vertical eye openness — the ratio that collapses when the eyelid shuts."""
    boxes = [eye_box(lm, e) for e in (L_EYE, R_EYE)]
    return sum(b["h"] / b["w"] for b in boxes) / 2


def ema(prev, value, factor=EMA):
    return factor * value + (1 - factor) * prev


class BlinkCounter:
    """Openness wave in, blink count out. Hysteresis keeps sensor noise from
    counting as blinks: a blink is a drop below CLOSED_BELOW followed by a
    recovery above OPEN_ABOVE."""

    def __init__(self):
        self.was_closed = False
        self.count = 0

    def update(self, value):
        blinked = False
        if not self.was_closed and value < CLOSED_BELOW:
            self.was_closed = True
        elif self.was_closed and value > OPEN_ABOVE:
            self.was_closed = False
            self.count += 1
            blinked = True
        return blinked


class SessionStats:
    """Accumulates one frame's verdicts, summarizes per minute."""

    def __init__(self):
        self.frames = 0
        self.off = 0
        self.blink_times = []   # seconds at which blinks completed

    def record(self, dx, dy, blinked, t):
        self.frames += 1
        if abs(dx) > OFF_CENTER or abs(dy) > OFF_CENTER:
            self.off += 1
        if blinked:
            self.blink_times.append(t)

    def summary(self, seconds):
        minutes = max(seconds / 60, 1e-9)
        off_pct = 100 * self.off / self.frames if self.frames else 0.0
        return {
            "frames": self.frames,
            "blinks": len(self.blink_times),
            "blinks_per_min": round(len(self.blink_times) / minutes, 1),
            "off_center_pct": round(off_pct, 1),
            "focus_score": round(max(0.0, 100 - off_pct)),
        }


def simulate(seconds, seed=7):
    """A deterministic synthetic session — same shapes a real face would give,
    so the metrics can be tested without a camera.

    Blink schedule: one blink every ~4s (15/min). Gaze: near the lens except a
    deliberate look-away every 20th frame for 6 frames.
    """
    import random

    rng = random.Random(seed)
    fps = 30
    openness_waves = deque()
    for i in range(int(seconds * fps)):
        # openness oscillates gently around 0.34, with a shut eyelid each blink
        blink = (i % (4 * fps)) < 3
        openness_waves.append(0.12 if blink else 0.34 + rng.uniform(-0.01, 0.01))
    return openness_waves, fps


def summarize_session(seconds, frame_iter, stats=None):
    """Drive SessionStats from any iterable of (openness, dx, dy) frames."""
    stats = stats or SessionStats()
    counter = BlinkCounter()
    smooth = (0.0, 0.0)
    for t, (val, dx, dy) in enumerate(frame_iter):
        smooth = (ema(smooth[0], dx), ema(smooth[1], dy))
        stats.record(smooth[0], smooth[1], counter.update(val), t / 30)
    return stats


def run_camera(seconds, csv_path=None):
    """Live webcam session. mediapipe and opencv are imported here, not at
    module level, so the core math (and the tests) run without them installed."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        print(f"error: camera mode needs the optional deps ({e.name}) — "
              f"pip install mediapipe opencv-python", file=sys.stderr)
        return 1

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    print("loading face model (~3 MB, once)…")
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=_model_path()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("error: no camera opened", file=sys.stderr)
        return 1

    stats = SessionStats()
    counter = BlinkCounter()
    smooth = (0.0, 0.0)
    t0 = cv2.getTickCount()
    print(f"watching for {seconds}s — look at the lens, then try looking away…")
    try:
        while (cv2.getTickCount() - t0) / cv2.getTickFrequency() < seconds:
            ok, frame = cam.read()
            if not ok:
                break
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(mp_img, cv2.getTickCount())
            t = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                dx, dy = gaze_offset(lm)
                smooth = (ema(smooth[0], dx), ema(smooth[1], dy))
                stats.record(smooth[0], smooth[1], counter.update(openness(lm)), t)
                print(f"\r{t:5.0f}s  blinks {len(stats.blink_times):3d}  "
                      f"off {stats.off:5d}  score {round(max(0, 100 - 100 * stats.off / stats.frames)):3d}",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        cam.release()
    print()
    _print_summary(stats.summary(seconds))
    if csv_path:
        _write_csv(stats, csv_path)
    return 0


def _print_summary(s):
    print(f"{s['frames']} frames  |  blinks {s['blinks']} ({s['blinks_per_min']}/min)  "
          f"|  off-center {s['off_center_pct']}%  |  focus score {s['focus_score']}/100")


def _write_csv(stats, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blink_seconds"])
        w.writerows([[t] for t in stats.blink_times])


def _model_path():
    """FaceLandmarker model, cached next to this file on first use."""
    import os
    import urllib.request

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task", dest)
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description="attention analytics from a webcam")
    p.add_argument("--seconds", type=int, default=60, help="session length (default 60)")
    p.add_argument("--csv", metavar="PATH", help="write blink timestamps to a CSV")
    p.add_argument("--simulate", action="store_true",
                   help="synthetic session instead of the camera (deterministic)")
    p.add_argument("--seed", type=int, default=7, help="simulate seed")
    args = p.parse_args(argv)

    if args.simulate:
        waves, fps = simulate(args.seconds, args.seed)
        frames = ((waves[i], 0.5, 0.02) if i % 20 >= 14 else (waves[i], 0.02, 0.01)
                  for i in range(len(waves)))
        stats = summarize_session(args.seconds, frames)
        _print_summary(stats.summary(args.seconds))
        if args.csv:
            _write_csv(stats, args.csv)
        return 0
    return run_camera(args.seconds, args.csv)


if __name__ == "__main__":
    sys.exit(main())
