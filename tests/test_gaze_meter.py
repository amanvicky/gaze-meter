"""Tests for gaze_meter. The core math and the simulate path need no camera,
no mediapipe, no opencv — that's the point of the lazy imports."""

import pytest

import gaze_meter as gm


def lm_point(x, y):
    return {"x": x, "y": y}


def make_lm(iris_dx=0.0, iris_dy=0.0, eye_w=0.10, eye_h=0.035):
    """A minimal face: both eyes centered on cx=0.5, iris displaced by iris_dx/dy."""
    lm = {}
    for eye in (gm.L_EYE, gm.R_EYE):
        lm[eye["outer"]] = lm_point(0.5 - eye_w / 2, 0.4)
        lm[eye["inner"]] = lm_point(0.5 + eye_w / 2, 0.4)
        lm[eye["top"]] = lm_point(0.5, 0.4 - eye_h / 2)
        lm[eye["bottom"]] = lm_point(0.5, 0.4 + eye_h / 2)
    for iris in (gm.L_IRIS, gm.R_IRIS):
        lm[iris] = lm_point(0.5 + iris_dx * eye_w, 0.4 + iris_dy * eye_w)
    return lm


def test_eye_box_geometry():
    box = gm.eye_box(make_lm(), gm.L_EYE)
    assert box["cx"] == pytest.approx(0.5)
    assert box["w"] == pytest.approx(0.10)
    assert box["h"] == pytest.approx(0.035)


def test_gaze_offset_centered_is_zero():
    dx, dy = gm.gaze_offset(make_lm())
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.0)


def test_gaze_offset_normalized_by_eye_width():
    # iris sits half an eye-width to the right -> dx = 0.5 regardless of eye size
    dx, _ = gm.gaze_offset(make_lm(iris_dx=0.5, eye_w=0.20))
    assert dx == pytest.approx(0.5)


def test_openness_is_height_over_width():
    assert gm.openness(make_lm(eye_w=0.10, eye_h=0.035)) == pytest.approx(0.35)


def test_blink_counter_counts_closed_open_cycles():
    c = gm.BlinkCounter()
    blinks = [c.update(v) for v in (0.34, 0.12, 0.10, 0.12, 0.34, 0.34)]
    assert blinks == [False, False, False, False, True, False]
    assert c.count == 1


def test_blink_counter_ignores_noise_between_blinks():
    c = gm.BlinkCounter()
    for v in (0.34, 0.34, 0.34):
        c.update(v)
    assert c.count == 0


def test_summary_math():
    s = gm.SessionStats()
    for i in range(300):                      # 10 seconds at 30 fps
        off = (i % 20) >= 14                  # 6 of every 20 frames away
        dx = 0.5 if off else 0.05
        s.record(dx, 0.02, blinked=False, t=i / 30)
    out = s.summary(10)
    assert out["frames"] == 300
    assert out["off_center_pct"] == 30.0
    assert out["focus_score"] == 70


def test_summary_empty_session():
    out = gm.SessionStats().summary(0)
    assert out["focus_score"] == 100 and out["off_center_pct"] == 0.0


def test_simulate_session_is_deterministic():
    a = gm.simulate(60, seed=7)
    b = gm.simulate(60, seed=7)
    assert list(a[0]) == list(b[0])


def test_simulate_end_to_end_metrics():
    waves, fps = gm.simulate(60, seed=7)
    frames = ((waves[i], 0.5, 0.02) if i % 20 >= 14 else (waves[i], 0.02, 0.01)
              for i in range(len(waves)))
    s = gm.summarize_session(60, frames).summary(60)
    # one blink every 4 s -> exactly 15/min (blink counting uses raw openness,
    # no smoothing). The off-center share lands near the pattern's 30 % minus
    # the EMA warm-up frames at each look-away edge, so assert a band here;
    # exact off-center math is covered by test_summary_math.
    assert s["frames"] == 1800
    assert s["blinks_per_min"] == 15.0
    assert 25.0 <= s["off_center_pct"] <= 32.0
    assert 65 <= s["focus_score"] <= 75


def test_ema_moves_toward_value():
    v = 0.0
    for _ in range(50):
        v = gm.ema(v, 1.0)
    assert v > 0.99


def test_main_simulate_prints_summary(capsys):
    assert gm.main(["--simulate", "--seconds", "30"]) == 0
    out = capsys.readouterr().out
    assert "focus score" in out and "blinks" in out


def test_module_imports_without_camera_deps():
    # gaze_meter must import (and its tests run) with mediapipe/cv2 absent
    import sys
    assert "cv2" not in sys.modules and "mediapipe" not in sys.modules
