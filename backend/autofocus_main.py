# autofocus_main.py
import time
import globals
from motioncontrols import move_relative
from autofocus_back import process_frame, detect_largest_object_square_roi
import porthandler


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def wait_motion_done(motion_platform):
    # Megvárja, amíg a firmware befejezi az összes mozgást
    porthandler.write(motion_platform, "M400")

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def acquire_frame(timeout_ms=2000):
    """
    Acquires a single frame from the camera with BGR8 conversion.
    
    Uses the unified grab_and_convert_frame() for consistent BayerGR10p -> BGR8 conversion.
    
    Args:
        timeout_ms: Timeout in milliseconds for frame retrieval
        
    Returns:
        frame_bgr: NumPy array (HxWx3, uint8, BGR format)
        
    Raises:
        RuntimeError: If camera not ready/open/grabbing or grab fails
    """
    cam = globals.camera
    if cam is None or not cam.IsOpen():
        raise RuntimeError("Camera not ready")

    lock = globals.grab_lock

    with lock:
        if not cam.IsGrabbing():
            raise RuntimeError("Camera is not grabbing. Stream must be running.")

        # Use unified grab+convert function
        from cameracontrol import grab_and_convert_frame
        frame_bgr = grab_and_convert_frame(cam, timeout_ms=timeout_ms)

    return frame_bgr


def move_to_virtual_z(motion_platform, current_z, target_z, settle_s=1):
    dz = float(target_z) - float(current_z)
    print('Menj' + str(dz) + 'pozira')
    if abs(dz) > 1e-9:
        move_relative(motion_platform, z=dz)

        # ✅ IDE KELL: várjuk meg a mozgás végét
        wait_motion_done(motion_platform)

        # opcionális: kis extra rezgéscsillapítás
        if settle_s and settle_s > 0:
            time.sleep(settle_s)

    return float(target_z)


def measure_focus_score(frame, roi):
    return process_frame(frame, roi=roi)


def measure_score(motion_platform, current_z, target_z, roi,  n_frames=1, timeout_ms=2000):
    """
    Elmegy target_z-re, készít n_frames képet, score-ol, és visszaadja (új_z, score).
    n_frames>1 esetén mediánt veszünk (zaj ellen).
    """
    current_z = move_to_virtual_z(motion_platform, current_z, target_z)

    scores = []
    n = max(1, int(n_frames))
    for _ in range(n):
        frame = acquire_frame(timeout_ms=timeout_ms)
        scores.append(measure_focus_score(frame, roi))

    scores.sort()
    score = scores[len(scores) // 2]  # median
    print('Scores: ' + str(scores))
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
    fine2_step=0.1,
    fine2_halfspan=0.5,

    # Measurement

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
    current_z = move_to_virtual_z(motion_platform, current_z, float(z_min))

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
            roi=roi,  n_frames=n_frames, timeout_ms=grab_timeout_ms
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
    print('Best z: ' + str(best_z))


    # -----------------------------------------------------------------
    # 2) FINE 1
    # -----------------------------------------------------------------
    fine1_targets = sorted(set(
        clamp(best_z + float(off), float(z_min), float(z_max))
        for off in fine1_offsets
    ))
    print('Fine1_targets_z:' + str(fine1_targets))
    fine1_points = []
    for zf in fine1_targets:
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi, n_frames=n_frames, timeout_ms=grab_timeout_ms
        )
        fine1_points.append((zf, sf))

    best1_z, best1_s = max(fine1_points, key=lambda p: p[1])
    print('Fine best z: ' + str(best1_z))
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
    print('Fine2 targets: ' + str(fine2_targets))
    fine2_points = []
    for zf in fine2_targets:
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi,  n_frames=n_frames, timeout_ms=grab_timeout_ms
        )
        fine2_points.append((zf, sf))

    final_z, final_s = max(fine2_points, key=lambda p: p[1])
    print('Final_z: ' + str(final_z))
    # opcionális: ráállunk a legjobb fókuszra
    if move_to_best:
        current_z = move_to_virtual_z(motion_platform, current_z, final_z)

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