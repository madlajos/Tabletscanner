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
