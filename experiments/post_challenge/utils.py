from cProfile import label
from turtle import color
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def draw_pred_contours(image, gts, segs, boxes, class_idx_list, clicks_cls=None, bboxs_from_clicks=None):
    # Handle single class_idx for backward compatibility
    if isinstance(class_idx_list, int):
        class_idx_list = [class_idx_list]
    
    num_classes = len(class_idx_list)
    
    # Create subplots in a row
    fig, axes = plt.subplots(1, num_classes, figsize=(10 * num_classes, 8))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    # Handle single subplot case
    if num_classes == 1:
        axes = [axes]
    
    for i, class_idx in enumerate(class_idx_list):
        ax = axes[i]
        
        # Show original image
        ax.imshow(image.mean(axis=0), cmap="gray")
        
        # Use contours instead of filled areas
        gt_mask = (gts == class_idx).mean(axis=0)
        pred_mask = (segs == class_idx).mean(axis=0)
        
        # Draw contours
        ax.contour((gt_mask > 0.00001).astype(float), levels=[0.5], colors='red', 
                  linewidths=2, linestyles='-', label='GT')
        ax.contour((pred_mask > 0.00001).astype(float), levels=[0.5], colors='blue', 
                  linewidths=2, linestyles='--', label='Predicted')
        
        # Add bounding box as rectangle
        if boxes is not None and len(boxes) >= class_idx:
            box_coords_1 = boxes[class_idx - 1]
            # Convert 3D coordinates to 2D for display
            y_min = box_coords_1["z_mid_y_min"]
            y_max = box_coords_1["z_mid_y_max"]
            x_min = box_coords_1["z_mid_x_min"]
            x_max = box_coords_1["z_mid_x_max"]
            
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor='green', facecolor='none',
                                   linestyle=':', label='BBox')
            ax.add_patch(rect)

        # Add clicks if available
        if clicks_cls is not None and len(clicks_cls) >= class_idx:
            clicks = clicks_cls[class_idx - 1]
            print(f"Click coordinates for class {class_idx}: {clicks}")
            
            # Track if we've added labels to avoid duplicates
            fg_labeled = False
            bg_labeled = False
            
            for c in clicks['fg']:
                label_fg = 'fg click' if not fg_labeled else None
                ax.plot(c[2], c[1], 'ro', markersize=5, label=label_fg, color='green')
                fg_labeled = True
                
            for c in clicks['bg']:
                label_bg = 'bg click' if not bg_labeled else None
                ax.plot(c[2], c[1], 'bo', markersize=5, label=label_bg, color='purple')
                bg_labeled = True
        
        if bboxs_from_clicks is not None and len(bboxs_from_clicks) >= class_idx:	
            bbox = bboxs_from_clicks[class_idx - 1]
            # Convert 3D coordinates to 2D for display
            y_min = bbox["z_mid_y_min"]
            y_max = bbox["z_mid_y_max"]
            x_min = bbox["z_mid_x_min"]
            x_max = bbox["z_mid_x_max"]
            
            rect_clicks = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                            linewidth=2, edgecolor='orange', facecolor='none',
                                            linestyle='--', label='BBox from Clicks')
            ax.add_patch(rect_clicks)
        
        ax.set_title(f"Class {class_idx}: GT (Red) vs Pred (Blue) vs Box (Green)")
        ax.legend()
    
    plt.tight_layout(pad=0.2)
    plt.show()

import numpy as np
from src.eval_metrics import ( 
    compute_edt,
    sample_coord,
)
import cc3d
def generate_clicks(gts, segs, clicks_cls, clicks_order):
    unique_gts = np.sort(np.unique(gts))
    for ind, cls in enumerate(sorted(unique_gts[1:])):
        if cls == 0:
            continue  # skip background

        segs_cls = (segs == cls).astype(
            np.uint8
        )  ### TODO : the segs are not defined yet
        gts_cls = (gts == cls).astype(np.uint8)

        # Compute error mask
        error_mask = (segs_cls != gts_cls).astype(np.uint8)
        if np.sum(error_mask) > 0:
            errors = cc3d.connected_components(
                error_mask, connectivity=26
            )  # 26 for 3D connectivity

            # Calculate the sizes of connected error components
            component_sizes = np.bincount(errors.flat)

            # Ignore non-error regions
            component_sizes[0] = 0

            # Find the largest error component
            largest_component_error = np.argmax(component_sizes)

            # Find the voxel coordinates of the largest error component
            largest_component = errors == largest_component_error

            edt = compute_edt(largest_component)
            center = sample_coord(edt)

            if (
                gts_cls[center] == 0
            ):  # oversegmentation -> place background click
                assert segs_cls[center] == 1
                clicks_cls[ind]["bg"].append(list(center))
                clicks_order[ind].append("bg")
            else:  # undersegmentation -> place foreground click
                assert segs_cls[center] == 0
                clicks_cls[ind]["fg"].append(list(center))
                clicks_order[ind].append("fg")

            assert largest_component[center]  # click within error
