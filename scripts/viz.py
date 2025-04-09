
from operator import gt
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

def visualize_slices_grid(image_path, gt_path, pred_path, n_slices=9, axis=0, output_dir=None, fig_title=None):
    """
    Visualizes a grid of n_slices sampled evenly from image, ground truth,
    and prediction arrays side-by-side.

    Args:
        image_path (str): Path to the .npy file containing the image data.
        gt_path (str): Path to the .npy file containing the ground truth data.
        pred_path (str): Path to the .npy file containing the prediction data.
        n_slices (int, optional): The number of slices to display in the grid. Defaults to 9.
        axis (int, optional): The axis along which to sample slices (0, 1, or 2). Defaults to 0.
        output_dir (str, optional): Directory to save the plot image. If None, displays the plot. Defaults to None.
        fig_title (str, optional): Custom title for the figure. Defaults to None.
    """
    print(f"Loading arrays:")
    print(f"  Image: {image_path}")
    print(f"  GT:    {gt_path}")
    print(f"  Pred:  {pred_path}")

    try:
        image = np.load(image_path, allow_pickle=True)['imgs']
        gt = np.load(gt_path, allow_pickle=True)['gts']
        pred = np.load(pred_path)["segs"]
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return
    except Exception as e:
        print(f"Error during file loading: {e}")
        return

    print(f"Shapes - Image: {image.shape}, GT: {gt.shape}, Pred: {pred.shape}")

    # --- Basic Sanity Checks ---
    if image.shape != gt.shape or image.shape != pred.shape:
        print("Warning: Array shapes do not match! Attempting to proceed but may fail.")
        # Check if slicing dimension is compatible
        if image.shape[axis] != gt.shape[axis] or image.shape[axis] != pred.shape[axis]:
             print(f"Error: Dimension size for axis {axis} does not match across arrays. Cannot proceed.")
             return

    if axis < 0 or axis >= image.ndim:
        print(f"Error: Invalid axis {axis}. Must be 0, 1, or 2 for 3D data.")
        return

    # Determine slice indices to sample evenly
    total_num_slices = image.shape[axis]
    if n_slices >= total_num_slices:
        # If requested slices > available, show all slices
        print(f"Warning: Requested {n_slices} slices >= available {total_num_slices}. Displaying all slices.")
        indices = np.arange(total_num_slices)
        n_slices = total_num_slices # Update n_slices for grid layout
    else:
        # Sample n_slices evenly, including start and end if possible
        indices = np.linspace(0, total_num_slices - 1, n_slices, dtype=int)

    print(f"Visualizing {n_slices} slices along axis {axis}. Indices: {indices}")

    # --- Determine grid layout ---
    # Aim for a layout close to square, preferring more columns
    ncols = 3 # One column for Image, one for GT, one for Pred
    nrows = n_slices
    fig_height = nrows * 3 # Adjust multiplier for desired slice height
    fig_width = ncols * 3  # Adjust multiplier for desired slice width

    # --- Create Plot ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False) # Ensure axes is always 2D

    if fig_title is None:
        base_name = os.path.splitext(os.path.basename(pred_path))[0]
        fig_title = f"Axis {axis} Slices Comparison - {base_name}"
    fig.suptitle(fig_title, fontsize=16)

    # --- Determine image intensity range for consistent display ---
    # Use percentiles across a sample of slices for robustness
    sample_indices_for_norm = np.linspace(0, total_num_slices - 1, min(10, total_num_slices), dtype=int)
    sample_slices = np.take(image, sample_indices_for_norm, axis=axis)
    vmin = np.percentile(sample_slices, 1)
    vmax = np.percentile(sample_slices, 99)

    # +++ DEBUGGING +++
    slice_min = sample_slices.min()
    slice_max = sample_slices.max()
    print(f"  Intensity Range Check:")
    print(f"    Slice Min: {slice_min}, Slice Max: {slice_max}")
    print(f"    Calculated vmin (1%): {vmin}, Calculated vmax (99%): {vmax}")
    # +++ END DEBUGGING +++

    # --- Add a safety check and fallback ---
    if vmin > vmax:
        print(f"  Warning: Calculated vmin ({vmin}) > vmax ({vmax}). Adjusting...")
        # Fallback strategy 1: Use min/max of the sample
        # vmin = slice_min
        # vmax = slice_max
        # Fallback strategy 2: If min == max, add a small epsilon for display
        if slice_min == slice_max:
             vmin = slice_min - 1 # Adjust slightly to create a valid range
             vmax = slice_max + 1
             print(f"    Slice is constant. Using adjusted range: vmin={vmin}, vmax={vmax}")
        else:
             # If range exists but percentiles failed, use min/max
             vmin = slice_min
             vmax = slice_max
             print(f"    Percentiles failed. Using full range: vmin={vmin}, vmax={vmax}")

    # Ensure vmin is strictly less than vmax if they became equal after fallback
    if vmin == vmax:
         print(f"  Warning: vmin == vmax after fallback ({vmin}). Adjusting slightly.")
         vmin = vmin - 1 # Or some small epsilon appropriate for your data type
         vmax = vmax + 1

    # --- Plot each sampled slice ---
    for i, slice_index in enumerate(indices):
        # Select Slices for this row
        img_slice = np.take(image, slice_index, axis=axis)
        gt_slice = np.take(gt, slice_index, axis=axis)
        pred_slice = np.take(pred, slice_index, axis=axis)

        # Plot Image Slice
        ax_img = axes[i, 0]
        ax_img.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax_img.set_title(f"Img Slice {slice_index}")
        ax_img.axis('off')

        # Plot Ground Truth Slice (Overlay)
        ax_gt = axes[i, 1]
        ax_gt.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
        ax_gt.imshow(gt_masked, cmap='jet', alpha=0.5, aspect='equal', vmin=1) # Use alpha, distinct cmap
        ax_gt.set_title(f"GT Slice {slice_index}")
        ax_gt.axis('off')

        # Plot Prediction Slice (Overlay)
        ax_pred = axes[i, 2]
        ax_pred.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
        ax_pred.imshow(pred_masked, cmap='jet', alpha=0.5, aspect='equal', vmin=1) # Use same settings
        ax_pred.set_title(f"Pred Slice {slice_index}")
        ax_pred.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # --- Save or Show Plot ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Use the figure title for the filename if available, else generate one
        filename_base = fig_title.replace(" ", "_").replace("/", "_").replace("\\", "_") if fig_title else os.path.splitext(os.path.basename(pred_path))[0]
        output_filename = os.path.join(output_dir, f"{filename_base}_axis{axis}_grid.png")
        try:
            plt.savefig(output_filename)
            print(f"Plot saved to: {output_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close(fig) # Close the figure after saving
    else:
        plt.show()

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a grid of slices from image, ground truth, and prediction NumPy arrays.")
    #parser.add_argument("image_path", help="Path to the image .npy file.")
    #parser.add_argument("gt_path", help="Path to the ground truth .npy file.")
    #parser.add_argument("pred_path", help="Path to the prediction .npy file.")
    parser.add_argument("-n", "--n_slices", type=int, default=9,
                        help="Number of slices to display in the grid (default: 9).")
    parser.add_argument("-a", "--axis", type=int, default=0, choices=[0, 1, 2],
                        help="Axis along which to slice (default: 0).")
    parser.add_argument("-o", "--output_dir", type=str, default="results/viz",
                        help="Directory to save the plot image (default: display plot).")
    parser.add_argument("-t", "--title", type=str, default=None,
                        help="Custom title for the figure.")


    import yaml
    config = yaml.safe_load(open("config.yaml"))
    print(os.listdir(config["RESULTS_DIR"]+"/sammed3d/"))
    
    file_name = os.listdir(config["RESULTS_DIR"]+"/sammed3d/")[0]
    file_name = os.path.splitext(file_name)[0]
    # remove the _pred at the end of the file
    file_name = file_name.replace("_pred", "")
    print(file_name)
    img_path = config["VAL_DIR"] + "/3D_val_npz/" + file_name + ".npz"
    img = np.load(img_path, allow_pickle=True)
    print(list(img.keys()))
    print(img['imgs'].shape) # 3d volume
    print(img['boxes']) # list of bbox coordinates, one per class
    print(img['spacing'])
    # print(img['text_prompts']) # dictionary of text propmts, one per class
    
    gt_path = config["VAL_DIR"] + "/3D_val_gt_interactive_seg/" + file_name + ".npz"
    gt = np.load(gt_path, allow_pickle=True)
    print(list(gt.keys())) # ['gts', 'boxes', 'spacing']
    print(gt['gts'].shape) # 3d volume
    print(gt['boxes']) # list of bbox coordinates, one per class
    print(gt['spacing']) # spacing of the image

    pred_path = config["RESULTS_DIR"] + "/sammed3d/" + file_name + "_pred.npz"
    pred = np.load(pred_path, allow_pickle=True)
    print(list(pred.keys())) # ['segs', 'all_segs']
    print(pred['segs'].shape) # 3d volume
    print(pred['all_segs'].shape) # 3d volumes

    args = parser.parse_args()
    for file_name in os.listdir(config["RESULTS_DIR"]+"/sammed3d/"):
        file_name = os.path.splitext(file_name)[0]
        # remove the _pred at the end of the file
        file_name = file_name.replace("_pred", "")

        args.image_path = config["VAL_DIR"] + "/3D_val_npz/" + file_name + ".npz"
        args.gt_path = config["VAL_DIR"] + "/3D_val_gt_interactive_seg/" + file_name + ".npz"
        args.pred_path = config["RESULTS_DIR"] + "/sammed3d/" + file_name + "_pred.npz"
        args.output_dir = config["RESULTS_DIR"] + "/viz/"

        visualize_slices_grid(args.image_path, args.gt_path, args.pred_path, args.n_slices, args.axis, args.output_dir, args.title)
