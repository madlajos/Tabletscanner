import cv2
import numpy as np


def flat_field_correction(
    data,
    method="downsampled",
    alpha=1.0,
    use_nlm=False,
    nlm_h=3.0,
    sigma_bg=80,
    down=0.10,
    sigma_small=8,
    final_blur=3,
    percentile_low=1,
    percentile_high=99,
    debug=False
):
    """
    Grayscale képen illumination / flat-field correction.

    method:
        - "gaussian": teljes képen nagy gaussian blur háttérhez
        - "downsampled": lekicsinyítés + blur + visszanagyítás

    alpha:
        háttérkorrekció erőssége

    use_nlm:
        fastNlMeansDenoising használata korrekció előtt

    nlm_h:
        NLM erősség

    sigma_bg:
        gaussian háttér blur sigma teljes képes módban

    down:
        lekicsinyítési arány a downsampled módszerhez

    sigma_small:
        blur sigma a kicsinyített képen

    final_blur:
        visszanagyított background simítása

    percentile_low / percentile_high:
        robusztus skálázás percentilisei
    """

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3201"
        return data

    if method not in ["gaussian", "downsampled"]:
        data["error"] = "E3202"
        return data

    if not isinstance(alpha, (int, float)) or alpha <= 0:
        data["error"] = "E3203"
        return data

    if not isinstance(nlm_h, (int, float)) or nlm_h < 0:
        data["error"] = "E3204"
        return data

    if not isinstance(sigma_bg, (int, float)) or sigma_bg <= 0:
        data["error"] = "E3205"
        return data

    if not isinstance(down, (int, float)) or down <= 0 or down > 1:
        data["error"] = "E3206"
        return data

    if not isinstance(sigma_small, (int, float)) or sigma_small <= 0:
        data["error"] = "E3207"
        return data

    if not isinstance(final_blur, (int, float)) or final_blur < 0:
        data["error"] = "E3208"
        return data

    if not isinstance(percentile_low, (int, float)) or not isinstance(percentile_high, (int, float)):
        data["error"] = "E3209"
        return data

    if percentile_low < 0 or percentile_high > 100 or percentile_low >= percentile_high:
        data["error"] = "E3210"
        return data

    output_images = []
    backgrounds = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3211"
            return data

        if len(img.shape) != 2:
            data["error"] = "E3212"
            return data

        # uint8 -> float32 [0,1]
        if img.dtype == np.uint8:
            I = img.astype(np.float32) / 255.0
        else:
            I = img.astype(np.float32)
            max_val = np.max(I) if I.size > 0 else 1.0
            if max_val > 1.0:
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
        else:
            I_f = I

        # háttérbecslés
        if method == "gaussian":
            bg = cv2.GaussianBlur(I_f, (0, 0), sigmaX=float(sigma_bg), sigmaY=float(sigma_bg))

        elif method == "downsampled":
            h, w = I_f.shape[:2]
            small_w = max(1, int(round(w * down)))
            small_h = max(1, int(round(h * down)))

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

        # átlag = 1 normalizálás
        I_corr = I_corr / (np.mean(I_corr) + 1e-12)

        # robusztus skálázás
        lo = np.percentile(I_corr, percentile_low)
        hi = np.percentile(I_corr, percentile_high)

        if hi <= lo:
            data["error"] = "E3213"
            return data

        I_corr = np.clip((I_corr - lo) / (hi - lo), 0.0, 1.0)

        # vissza uint8-ra, hogy kompatibilis maradjon a további pipeline-nal
        out = np.clip(I_corr * 255.0, 0, 255).astype(np.uint8)
        bg_u8 = np.clip(bg / np.max(bg) * 255.0, 0, 255).astype(np.uint8)

        output_images.append(out)
        backgrounds.append(bg_u8)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["results"]["background_images"] = backgrounds
    data["meta"]["flat_field_correction"] = {
        "method": method,
        "alpha": float(alpha),
        "use_nlm": bool(use_nlm),
        "nlm_h": float(nlm_h),
        "sigma_bg": float(sigma_bg),
        "down": float(down),
        "sigma_small": float(sigma_small),
        "final_blur": float(final_blur),
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high)
    }
    data["history"].append("flat_field_correction")

    if debug:
        print(data["meta"]["flat_field_correction"])
        cv2.imshow("flat_field_input_bg_output", np.hstack([data["results"]["background_images"][0], output_images[0]]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return data
