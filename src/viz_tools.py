

import matplotlib.pyplot as plt
import os 
import numpy as np
from math import ceil, sqrt

def plot_middle_slice(volume_data, title="Middle Slice", slice_dim=0, cmap='gray', save_dir="debug_plots"):    
    print("saucdde")
    """
    Plots the middle slice of a 3D NumPy array using Matplotlib.

    Args:
        volume_data (np.ndarray): The 3D NumPy array (e.g., GT mask or image).
                                  Assumes Z, Y, X order if slice_dim=0.
        title (str): The title for the plot.
        slice_dim (int): The dimension along which to take the middle slice (0 for Z, 1 for Y, 2 for X).
        cmap (str): The colormap to use for imshow.
    """
    import numpy as np


    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True) # Create dir if needed

    middle_slice_idx = volume_data.shape[slice_dim] // 2
    middle_slice = np.take(volume_data, middle_slice_idx, axis=slice_dim)

    print(f"Plotting '{title}': Slice index {middle_slice_idx} along axis {slice_dim}. Shape: {middle_slice.shape}")
    print(f"Data range in slice: min={np.min(middle_slice):.2f}, max={np.max(middle_slice):.2f}, unique values: {np.unique(middle_slice)}")

    fig, ax = plt.subplots(figsize=(6, 6)) # Get figure and axes objects
    im = ax.imshow(middle_slice, cmap=cmap, origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_title(title + f" (Slice {middle_slice_idx} along axis {slice_dim})")
    ax.set_xlabel(f"Axis {(slice_dim + 2) % 3}")
    ax.set_ylabel(f"Axis {(slice_dim + 1) % 3}")
    fig.tight_layout()

    # --- SAVE INSTEAD OF SHOW ---
    # Sanitize title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_')
    save_path = os.path.join(save_dir, f"{safe_title}_slice_{slice_dim}_{middle_slice_idx}.png")
    """
    try:
       fig.savefig(save_path)
        print(f"Saved plot to: {save_path}")
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
    plt.close(fig) # Close the figure to free memory, important in loops!
    """
    plt.show() # Remove or comment out plt.show()


def center_of_mass(array):
    """
    Calculate the center of mass of a 3D numpy array.
    
    Parameters:
    array (numpy.ndarray): 3D array where values represent mass or intensity
    
    Returns:
    tuple: (z, y, x) coordinates of the center of mass
    """
    # Get array dimensions
    z_dim, y_dim, x_dim = array.shape
    
    # Create coordinate arrays
    z_coords = np.arange(z_dim)
    y_coords = np.arange(y_dim)
    x_coords = np.arange(x_dim)
    
    # Calculate total mass/intensity
    total_mass = np.sum(array)
    
    # Handle the case where total_mass is zero
    if total_mass == 0:
        return (z_dim / 2, y_dim / 2, x_dim / 2)  # Return center of array
    
    # Calculate center of mass for each dimension
    z_center = np.sum(z_coords[:, np.newaxis, np.newaxis] * array) / total_mass
    y_center = np.sum(y_coords[np.newaxis, :, np.newaxis] * array) / total_mass
    x_center = np.sum(x_coords[np.newaxis, np.newaxis, :] * array) / total_mass
    
    return (z_center, y_center, x_center)

def save_volume_viz(img_array, save_path=None, slice_indices=None, show=False):
    """
    Save volume slices to a single figure at the specified path.
    
    Parameters:
    -----------
    img_array : ndarray
        3D array representing the volume data, expected shape (D, H, W) where D is depth
    save_path : str or Path
        Path where to save the visualization figure
    slice_indices : None, int, list, or tuple, optional
        - None: Uses the middle slice
        - int: Uses the specified slice index
        - list/tuple of ints: Uses the specified slices
        - tuple of (int, 'uniform'): Uses evenly spaced slices across the volume
    
    Examples:
    ---------
    # Save middle slice
    save_volume_viz(volume, "middle_slice.png")
    
    # Save specific slice
    save_volume_viz(volume, "slice_10.png", 10)
    
    # Save multiple specific slices in one figure
    save_volume_viz(volume, "multiple_slices.png", [10, 20, 30])
    
    # Save 5 uniformly distributed slices in one figure
    save_volume_viz(volume, "uniform_slices.png", (5, 'uniform'))
    """

    depth = img_array.shape[0]
    
    # Determine which slices to save
    if slice_indices is None:
        slices = [depth // 2]  # Middle slice
    elif isinstance(slice_indices, int):
        slices = [slice_indices]
    elif isinstance(slice_indices, (list, tuple)):
        if len(slice_indices) == 2 and slice_indices[1] == 'uniform':
            # Uniform spacing
            num = slice_indices[0]
            slices = np.linspace(0, depth-1, num, dtype=int).tolist() if num > 1 else [depth // 2]
        else:
            # List of indices
            slices = slice_indices
    
    # Filter out any out-of-bounds indices
    slices = [idx for idx in slices if 0 <= idx < depth]
    
    # Create a single figure with subplots
    n = len(slices)
    if n == 1:
        plt.figure(figsize=(10, 8))
        plt.imshow(img_array[slices[0]], cmap='gray')
        plt.axis('off')
    else:
        # Determine grid size
        cols = min(4, ceil(sqrt(n)))
        rows = ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten() if n > 1 else [axes]
        
        # Plot each slice
        for i, idx in enumerate(slices):
            if i < len(axes):
                axes[i].imshow(img_array[idx], cmap='gray')
                axes[i].set_title(f"Slice {idx}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
    if show:
        plt.show()
    # Save the figure
    else:
        plt.savefig(save_path)
    plt.close()

