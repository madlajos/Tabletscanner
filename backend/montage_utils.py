"""
Image montage generation utility for batch preview.
Creates grid montages of multiple images with labels.
"""

import cv2
import numpy as np
import os


def calculate_grid_layout(num_images):
    """
    Calculate optimal grid layout for N images.
    Returns (rows, cols) tuple to distribute images evenly.
    """
    if num_images <= 1:
        return (1, 1)
    if num_images <= 2:
        return (1, 2)
    if num_images <= 4:
        return (2, 2)
    if num_images <= 6:
        return (2, 3)
    if num_images <= 9:
        return (3, 3)
    if num_images <= 12:
        return (3, 4)
    if num_images <= 16:
        return (4, 4)
    
    # For larger numbers, use square root
    side = int(np.ceil(np.sqrt(num_images)))
    return (side, side)


def load_image_safe(image_path):
    """Load image from path, return None if fails."""
    if not os.path.exists(image_path):
        return None
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return img
    except Exception:
        return None


def create_montage(image_paths, target_cell_width=200, target_cell_height=200, label_height=30, debug=False):
    """
    Create a montage grid from multiple images with labels.
    
    Args:
        image_paths: list of image file paths
        target_cell_width: width of each cell in montage (pixels)
        target_cell_height: height of each cell in montage (pixels)
        label_height: height reserved for label at bottom of each cell
        debug: print debug info
    
    Returns:
        montage_image: numpy array (BGR image) or None if fails
    """
    
    if not image_paths or len(image_paths) == 0:
        return None
    
    # Calculate layout
    rows, cols = calculate_grid_layout(len(image_paths))
    
    # Cell dimensions (image + label)
    cell_height = target_cell_height + label_height
    cell_width = target_cell_width
    
    # Total montage size
    montage_height = rows * cell_height
    montage_width = cols * cell_width
    
    # Create blank montage (white background)
    montage = np.ones((montage_height, montage_width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 0, 0)  # Black text
    
    for idx, img_path in enumerate(image_paths):
        # Calculate grid position
        row = idx // cols
        col = idx % cols
        
        # Top-left corner of this cell
        y_start = row * cell_height
        x_start = col * cell_width
        
        # Image region in cell (top part)
        y_img_end = y_start + target_cell_height
        x_img_end = x_start + target_cell_width
        
        # Label region (bottom part)
        y_label_start = y_img_end
        y_label_end = y_label_start + label_height
        
        # Load image
        img = load_image_safe(img_path)
        
        if img is not None:
            # Resize to fit cell (maintain aspect ratio)
            h, w = img.shape[:2]
            scale = min(target_cell_width / w, target_cell_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Center in cell
            y_offset = (target_cell_height - new_h) // 2
            x_offset = (target_cell_width - new_w) // 2
            
            y_img_start = y_start + y_offset
            y_img_fin = y_img_start + new_h
            x_img_start = x_start + x_offset
            x_img_fin = x_img_start + new_w
            
            # Place image in montage
            montage[y_img_start:y_img_fin, x_img_start:x_img_fin] = img_resized
        else:
            # Draw X for missing/failed images
            p1 = (x_start + 10, y_start + 10)
            p2 = (x_img_end - 10, y_img_end - 10)
            cv2.line(montage, p1, p2, (0, 0, 255), 2)  # Red X
            cv2.line(montage, (p2[0], p1[1]), (p1[0], p2[1]), (0, 0, 255), 2)
        
        # Draw label with image number
        label_text = f"Image {idx + 1}"
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = x_start + (cell_width - text_size[0]) // 2
        text_y = y_label_start + (label_height + text_size[1]) // 2
        
        # Draw white background for text
        cv2.rectangle(montage, 
                     (x_start, y_label_start),
                     (x_img_end, y_label_end),
                     (255, 255, 255),  # White
                     -1)
        
        # Draw text
        cv2.putText(montage, label_text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        
        # Draw cell border
        cv2.rectangle(montage, (x_start, y_start), (x_img_end, y_label_end), (200, 200, 200), 1)
    
    if debug:
        print(f"Montage created: {len(image_paths)} images, grid {rows}x{cols}, size {montage_width}x{montage_height}")
    
    return montage
