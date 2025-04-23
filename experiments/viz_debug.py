import napari
import numpy as np
import os
import argparse
from glob import glob

def visualize_arrays(file_paths):
    """
    Loads multiple .npy files and displays them as layers in napari.

    Args:
        file_paths (list): A list of paths to the .npy files to load.
    """
    if not file_paths:
        print("No file paths provided for visualization.")
        return

    viewer = napari.Viewer()
    print("Loading arrays into napari viewer...")

    for fpath in file_paths:
        try:
            # Extract a meaningful name from the filename
            base_name = os.path.basename(fpath).replace('.npy', '')
            print(f"  Loading: {fpath} as layer '{base_name}'")

            # Load the array
            array_data = np.load(fpath)

            # Add to viewer - adjust colormap/contrast as needed
            if 'image' in base_name.lower():
                viewer.add_image(array_data, name=base_name, colormap='gray', contrast_limits=(np.min(array_data), np.max(array_data)))
            elif 'label' in base_name.lower() or 'gt' in base_name.lower():
                 # Use labels layer for integer masks, easier to see distinct regions
                 viewer.add_labels(array_data, name=base_name)
            elif 'pred' in base_name.lower() or 'mask' in base_name.lower() or 'segmentation' in base_name.lower():
                 # Use labels layer for prediction masks too
                 viewer.add_labels(array_data, name=base_name)
            else:
                 # Default to image layer if type is unclear
                 viewer.add_image(array_data, name=base_name, colormap='gray', contrast_limits=(np.min(array_data), np.max(array_data)))

        except FileNotFoundError:
            print(f"  Warning: File not found - {fpath}")
        except Exception as e:
            print(f"  Error loading or adding file {fpath}: {e}")

    if viewer.layers:
        print("\nNapari viewer opened. Close the viewer window to exit the script.")
        napari.run() # Blocks until the viewer is closed
    else:
        print("No layers were successfully added to the viewer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved NumPy arrays from SAM debugging using Napari.")
    parser.add_argument(
        'file_patterns',
        nargs='+', # Allows one or more file paths/patterns
        help="Paths or glob patterns to the .npy files to visualize (e.g., 'debug_dir/*.npy', 'debug_dir/debug_class1_*.npy')."
    )

    args = parser.parse_args()

    all_files_to_load = []
    for pattern in args.file_patterns:
        found_files = sorted(glob(pattern)) # Use glob to handle wildcards and sort
        if not found_files:
             print(f"Warning: No files found matching pattern: {pattern}")
        all_files_to_load.extend(found_files)

    # Remove duplicates if patterns overlap
    unique_files = sorted(list(set(all_files_to_load)))

    if unique_files:
        visualize_arrays(unique_files)
    else:
        print("No valid .npy files found based on the provided patterns.")
