# autofocus_main.py
import time
import globals
from motioncontrols import move_relative
from autofocus_back import process_frame, detect_largest_object_square_roi
from cameracontrol import converter


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def acquire_frame(timeout_ms=2000):
    from pypylon import pylon

    cam = globals.camera
    if cam is None or not cam.IsOpen():
        raise RuntimeError("Camera not ready")

    lock = globals.grab_lock

    with lock:
        if not cam.IsGrabbing():
            raise RuntimeError("Camera is not grabbing. Stream must be running.")

        grab_result = cam.RetrieveResult(int(timeout_ms), pylon.TimeoutHandling_ThrowException)
        try:
            if not grab_result.GrabSucceeded():
                raise RuntimeError("Grab failed")

            # Convert BayerGR10p (or whatever the camera outputs) -> BGR8 for OpenCV
            frame_bgr = converter.Convert(grab_result).GetArray()

            # If you need to keep it beyond this scope, copy it (safe):
            frame_bgr = frame_bgr.copy()

        finally:
            grab_result.Release()

    return frame_bgr


def move_to_virtual_z(motion_platform, current_z, target_z, settle_s=3):
    """
    Relatív lépéssel odaáll target_z-re (virtuális Z), és visszaadja az új current_z-t.
    """
    dz = 1
    if abs(dz) > 1e-9:
        move_relative(motion_platform, z=dz)
        if settle_s > 0:
            time.sleep(settle_s)
    return target_z


def measure_focus_score(frame, roi):
    return process_frame(frame, roi=roi)


def measure_score(motion_platform, current_z, target_z, roi, settle_s=0.0, n_frames=1, timeout_ms=2000):
    """
    Elmegy target_z-re, készít n_frames képet, score-ol, és visszaadja (új_z, score).
    n_frames>1 esetén mediánt veszünk (zaj ellen).
    """
    current_z = move_to_virtual_z(motion_platform, current_z, target_z, settle_s=settle_s)

    scores = []
    n = max(1, int(n_frames))
    for _ in range(n):
        frame = acquire_frame(timeout_ms=timeout_ms)
        scores.append(measure_focus_score(frame, roi))

    scores.sort()
    score = scores[len(scores) // 2]  # median
    return current_z, score


# ---------------------------------------------------------------------
# Autofocus main
# ---------------------------------------------------------------------

def autofocus_coarse(
    motion_platform,
    z_min=0.0,
    z_max=25.0,

    # Coarse scan
    coarse_step=2.0,
    drop_ratio=0.92,     # stop, ha score < best*drop_ratio két ponton át a best után
    bad_needed=2,
    min_points=4,

    # Fine 1 around coarse best
    fine1_offsets=(-1.0, -0.5, 0.0, +0.5, +1.0),

    # Fine 2 around fine1 best
    fine2_step=0.25,
    fine2_halfspan=0.25,

    # Measurement
    settle_s=0.5,
    n_frames=1,
    roi_square_scale=0.8,
    grab_timeout_ms=2000,

    # Return / behavior
    move_to_best=True,
):
    """
    Determinisztikus AF (relatív Z tartományban):
      1) coarse scan (coarse_step)
      2) fine1 a coarse best körül (fine1_offsets)
      3) fine2 a fine1 best körül (±fine2_halfspan, fine2_step)

    ROI-t csak egyszer detektálunk az első frame-en, utána fixen használjuk.

    VISSZATÉRÉS:
      dict, benne a 'focus_pos' (relatív Z), és ha ismert a 'focus_pos_abs_z'.
    """

    # Virtuális Z nyilvántartás (0 a híváskori Z-hez képest)
    current_z = 0.0
    current_z = move_to_virtual_z(motion_platform, current_z, float(z_min), settle_s=settle_s)

    # --- ROI detektálás 1x ---
    first_frame = acquire_frame(timeout_ms=grab_timeout_ms)

    roi = detect_largest_object_square_roi(
        first_frame,
        square_scale=roi_square_scale,
        debug_scale=0.3,
        show_debug=False
    )
    if roi is None:
        roi = None  # fallback: teljes kép

    # -----------------------------------------------------------------
    # 1) COARSE SCAN
    # -----------------------------------------------------------------
    coarse_points = []  # (z, score)
    best_z = None
    best_score = float("-inf")
    bad_count = 0

    z = float(z_min)
    i = 0
    while z <= float(z_max):
        current_z, s = measure_score(
            motion_platform, current_z, z,
            roi=roi, settle_s=settle_s, n_frames=n_frames, timeout_ms=grab_timeout_ms
        )
        coarse_points.append((z, s))

        if s > best_score:
            best_score = s
            best_z = z
            bad_count = 0

        if (i + 1) >= int(min_points) and (best_z is not None) and (z > best_z):
            if s < best_score * float(drop_ratio):
                bad_count += 1
            else:
                bad_count = 0
            if bad_count >= int(bad_needed):
                break

        z += float(coarse_step)
        i += 1

    if best_z is None:
        raise RuntimeError("Coarse scan nem adott best_z-t.")

    # --- Coarse érvényesség ellenőrzés: legyen elég kontraszt a score-ok között ---
    if len(coarse_points) >= 2:
        scores_only = [s for _, s in coarse_points]
        score_min = min(scores_only)
        score_max = max(scores_only)
        if (score_max - score_min) <= 0.2:
            raise RuntimeError(
                f"Coarse scan érvénytelen: score_max-score_min={score_max - score_min:.4f} <= 0.2 "
                f"(max={score_max:.4f}, min={score_min:.4f})."
            )
    else:
        raise RuntimeError("Coarse scan érvénytelen: túl kevés mérési pont.")

    # -----------------------------------------------------------------
    # 2) FINE 1
    # -----------------------------------------------------------------
    fine1_targets = sorted(set(
        clamp(best_z + float(off), float(z_min), float(z_max))
        for off in fine1_offsets
    ))

    fine1_points = []
    for zf in fine1_targets:
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi, settle_s=settle_s, n_frames=n_frames, timeout_ms=grab_timeout_ms
        )
        fine1_points.append((zf, sf))

    best1_z, best1_s = max(fine1_points, key=lambda p: p[1])

    # -----------------------------------------------------------------
    # 3) FINE 2
    # -----------------------------------------------------------------
    halfspan = float(fine2_halfspan)
    step2 = float(fine2_step)
    if step2 <= 0:
        raise ValueError("fine2_step legyen > 0.")

    k = int(round(halfspan / step2))
    offsets2 = [j * step2 for j in range(-k, k + 1)]

    fine2_targets = sorted(set(
        clamp(best1_z + off, float(z_min), float(z_max))
        for off in offsets2
    ))

    fine2_points = []
    for zf in fine2_targets:
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi, settle_s=settle_s, n_frames=n_frames, timeout_ms=grab_timeout_ms
        )
        fine2_points.append((zf, sf))

    final_z, final_s = max(fine2_points, key=lambda p: p[1])

    # opcionális: ráállunk a legjobb fókuszra
    if move_to_best:
        current_z = move_to_virtual_z(motion_platform, current_z, final_z, settle_s=settle_s)

    # abszolút z (ha a rendszered cache-eli valahol)
    abs_z0 = None
    try:
        abs_z0 = globals.last_toolhead_pos.get("z", None)
        if abs_z0 is not None:
            abs_z0 = float(abs_z0)
    except Exception:
        abs_z0 = None

    result = {
        "roi": roi,
        "coarse_points": coarse_points,
        "best_coarse": (best_z, best_score),
        "fine1_points": fine1_points,
        "best_fine1": (best1_z, best1_s),
        "fine2_points": fine2_points,
        "final": (final_z, final_s),

        # EZT KÉRTED: a fókusz pozíció
        "focus_pos": {"z_rel": float(final_z), "score": float(final_s)},
    }

    if abs_z0 is not None:
        result["focus_pos"]["z_abs_est"] = abs_z0 + float(final_z)

    return {"status": "OK", "z_rel": float(final_z), "score": float(final_s)}
