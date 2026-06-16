"""
Processing elements for the image analysis pipeline.
Each element operates on the shared data dict:
    {
        "images": [np.ndarray, ...],
        "paths": [str, ...],
        "count": int,
        "meta": dict,
        "results": dict,
        "history": [str, ...],
        "error": str | None,
    }
"""
from proc_elements.load_img import create_data, load_image
from proc_elements.select_channel import select_channel
from proc_elements.apply_thresh import apply_threshold
from proc_elements.generate_histogram import calculate_histograms
from proc_elements.range_mask import apply_range_mask
from proc_elements.calc_intensity import calculate_intensity_stats
from proc_elements.add_measured import add_sequence_values
from proc_elements.curve_fitting import fit_curve
from proc_elements.pred_from_int import predict_node
from proc_elements.apply_blur import apply_blur
from proc_elements.histogram_eq import histogram_equalization
from proc_elements.clahe import apply_clahe
from proc_elements.normalization import normalize_images
from proc_elements.bright_contr import adjust_brightness_contrast
from proc_elements.gamma_corr import gamma_correction
from proc_elements.flat_field_corr import flat_field_correction
from proc_elements.robust_stretch import robust_stretch_gamma
from proc_elements.advanced_ill_corr import advanced_illumin_corr
from proc_elements.draw_roi import mask_roi, mask_rect_roi
from proc_elements.scale_bar import scale_bar_overlay
from proc_elements.resize_img import resize_images
from proc_elements.region_attr import detect_particles
from proc_elements.create_pca import histogram_pca
from proc_elements.detect_circ import detect_circles
from proc_elements.gray_map import fixed_centroid_soft_hard_node as gray_map
from proc_elements.rgb_gray_map import rgb_gray_map_node as rgb_gray_map
from proc_elements.dual_map import dual_map_node as dual_map
from proc_elements.cls_like import rgb_cls_reference_mapping_node as rgb_cls_reference_mapping
from proc_elements.store_gray_images import store_as_gray_images
from proc_elements.filter_region import characterize_particles
from proc_elements.color_thresh import color_threshold
