

import matplotlib.pyplot as plt
import numpy as np
import os 

def plot_middle_slice(volume_data, title="Middle Slice", slice_dim=0, cmap='gray', save_dir="debug_plots"):    
    """
    Plots the middle slice of a 3D NumPy array using Matplotlib.

    Args:
        volume_data (np.ndarray): The 3D NumPy array (e.g., GT mask or image).
                                  Assumes Z, Y, X order if slice_dim=0.
        title (str): The title for the plot.
        slice_dim (int): The dimension along which to take the middle slice (0 for Z, 1 for Y, 2 for X).
        cmap (str): The colormap to use for imshow.
    """

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
    try:
        fig.savefig(save_path)
        print(f"Saved plot to: {save_path}")
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
    plt.close(fig) # Close the figure to free memory, important in loops!
    # plt.show() # Remove or comment out plt.show()

    import numpy as np

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