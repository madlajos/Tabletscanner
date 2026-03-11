import cv2
import numpy as np


def advanced_illumin_corr(
    data,
    bg_method="downsampled",
    alpha=1.0,
    use_nlm=False,
    nlm_h=3.0,
    sigma_bg=80,
    down=0.10,
    sigma_small=8,
    final_blur=3,
    percentile_low=1,
    percentile_high=99,
    use_gamma=False,
    gamma=1.0,
    debug=False
):
    """
    Advanced illumination correction grayscale képekre.

    Lépések:
    1) opcionális NLM denoise
    2) background / illumination field becslés
    3) flat-field correction
    4) mean normalization
    5) robust percentile scaling
    6) opcionális gamma correction

    bg_method:
        - "gaussian"
        - "downsampled"
    """

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3301"
        return data

    if bg_method not in ["gaussian", "downsampled"]:
        data["error"] = "E3302"
        return data

    if not isinstance(alpha, (int, float)) or alpha <= 0:
        data["error"] = "E3303"
        return data

    if not isinstance(use_nlm, bool):
        data["error"] = "E3304"
        return data

    if not isinstance(nlm_h, (int, float)) or nlm_h < 0:
        data["error"] = "E3305"
        return data

    if not isinstance(sigma_bg, (int, float)) or sigma_bg <= 0:
        data["error"] = "E3306"
        return data

    if not isinstance(down, (int, float)) or down <= 0 or down > 1:
        data["error"] = "E3307"
        return data

    if not isinstance(sigma_small, (int, float)) or sigma_small <= 0:
        data["error"] = "E3308"
        return data

    if not isinstance(final_blur, (int, float)) or final_blur < 0:
        data["error"] = "E3309"
        return data

    if not isinstance(percentile_low, (int, float)) or not isinstance(percentile_high, (int, float)):
        data["error"] = "E3310"
        return data

    if percentile_low < 0 or percentile_high > 100 or percentile_low >= percentile_high:
        data["error"] = "E3311"
        return data

    if not isinstance(use_gamma, bool):
        data["error"] = "E3312"
        return data

    if not isinstance(gamma, (int, float)) or gamma <= 0:
        data["error"] = "E3313"
        return data

    output_images = []
    background_images = []
    denoised_images = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3314"
            return data

        if len(img.shape) != 2:
            data["error"] = "E3315"
            return data

        # float [0..1]
        if img.dtype == np.uint8:
            I = img.astype(np.float32) / 255.0
        else:
            I = img.astype(np.float32)
            if I.size == 0:
                data["error"] = "E3316"
                return data

            if np.max(I) > 1.0:
                I = I / 255.0

        # opcionális NLM
        if use_nlm:
            I_u8 = np.clip(I * 255.0, 0, 255).astype(np.uint8)
            I_f_u8 = cv2.fastNlMeansDenoising(
                I_u8,
                None,
                h=float(nlm_h),
                templateWindowSize=7,
                searchWindowSize=21
            )
            I_f = I_f_u8.astype(np.float32) / 255.0
            denoised_images.append(I_f_u8)
        else:
            I_f = I
            denoised_images.append(np.clip(I_f * 255.0, 0, 255).astype(np.uint8))

        # háttérbecslés
        if bg_method == "gaussian":
            bg = cv2.GaussianBlur(
                I_f,
                (0, 0),
                sigmaX=float(sigma_bg),
                sigmaY=float(sigma_bg)
            )

        elif bg_method == "downsampled":
            h, w = I_f.shape[:2]
            small_w = max(1, int(round(w * float(down))))
            small_h = max(1, int(round(h * float(down))))

            I_small = cv2.resize(I_f, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

            bg_small = cv2.GaussianBlur(
                I_small,
                (0, 0),
                sigmaX=float(sigma_small),
                sigmaY=float(sigma_small)
            )

            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

            if final_blur > 0:
                bg = cv2.GaussianBlur(
                    bg,
                    (0, 0),
                    sigmaX=float(final_blur),
                    sigmaY=float(final_blur)
                )

        bg = bg + 1e-6

        # flat-field correction
        I_corr = I_f / (bg ** float(alpha))

        # mean normalization
        I_corr = I_corr / (np.mean(I_corr) + 1e-12)

        # robust scaling
        lo = np.percentile(I_corr, percentile_low)
        hi = np.percentile(I_corr, percentile_high)

        if hi <= lo:
            data["error"] = "E3317"
            return data

        I_corr = np.clip((I_corr - lo) / (hi - lo), 0.0, 1.0)

        # opcionális gamma
        if use_gamma:
            I_corr = np.clip(I_corr ** float(gamma), 0.0, 1.0)

        out = np.clip(I_corr * 255.0, 0, 255).astype(np.uint8)

        bg_norm = bg / (np.max(bg) + 1e-12)
        bg_u8 = np.clip(bg_norm * 255.0, 0, 255).astype(np.uint8)

        output_images.append(out)
        background_images.append(bg_u8)

    data["images"] = output_images
    data["count"] = len(output_images)

    data["results"]["advanced_illum_backgrounds"] = background_images
    data["results"]["advanced_illum_denoised"] = denoised_images

    data["meta"]["advanced_illumin_corr"] = {
        "bg_method": bg_method,
        "alpha": float(alpha),
        "use_nlm": bool(use_nlm),
        "nlm_h": float(nlm_h),
        "sigma_bg": float(sigma_bg),
        "down": float(down),
        "sigma_small": float(sigma_small),
        "final_blur": float(final_blur),
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high),
        "use_gamma": bool(use_gamma),
        "gamma": float(gamma)
    }

    data["history"].append("advanced_illumin_corr")

    if debug:
        print(data["meta"]["advanced_illumin_corr"])
        print(f"Processed images: {len(output_images)}")
        cv2.imshow("advanced_illum_output", output_images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return data
