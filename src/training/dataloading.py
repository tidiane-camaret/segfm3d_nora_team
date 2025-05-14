import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleInteractiveSegmentationDataset(Dataset):
    """
    Simple dataset for interactive medical image segmentation using bounding boxes.
    """
    def __init__(self, 
                 img_dir,
                 gt_dir,
                 use_full_volume=True,  # Use full 3D volume or middle slice
                 val_ratio=0.2,         # Ratio of validation split
                 split='train',         # 'train' or 'val'
                 seed=42):              # Random seed for reproducibility
        """
        Args:
            img_dir: Directory containing image .npz files
            gt_dir: Directory containing ground truth .npz files
            use_full_volume: Whether to use full 3D volume (vs middle slice)
            val_ratio: Ratio of data to use for validation
            split: 'train' or 'val'
            seed: Random seed for reproducibility
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.use_full_volume = use_full_volume
        self.split = split
        
        # Get all .npz files in the image directory
        all_cases = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Split into train and validation sets
        n_val = int(len(all_cases) * val_ratio)
        if n_val == 0 and val_ratio > 0:
            n_val = 1  # Ensure at least one validation case if possible
            
        # Shuffle the cases deterministically
        random.shuffle(all_cases)
        
        if split == 'train':
            self.cases = all_cases[n_val:]
        else:  # val
            self.cases = all_cases[:n_val]
        
        print(f"Loaded {len(self.cases)} cases for {split} split")
        
    def __len__(self):
        return len(self.cases)
    
    def _normalize_image(self, image):
        """Normalize image values to [0,1] range"""
        # Find min and max per image to handle potential outliers
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        
        return image.astype(np.float32)
    
    def _create_bbox_channel(self, shape, bbox):
        """Create a binary mask channel for the bounding box"""
        bbox_channel = np.zeros(shape, dtype=np.float32)
        
        # Extract bbox coordinates
        #print(bbox.keys())
        z_min, z_max = bbox['z_min'], bbox['z_max']
        try:
            y_min, y_max = bbox['z_mid_y_min'], bbox['z_mid_y_max']
            x_min, x_max = bbox['z_mid_x_min'], bbox['z_mid_x_max']
        except KeyError:
            y_min, y_max = bbox['y_min'], bbox['y_max']
            x_min, x_max = bbox['x_min'], bbox['x_max']
        
        # Set the region inside the bounding box to 1
        bbox_channel[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = 1.0
        
        return bbox_channel
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        case_filename = self.cases[idx]
        case_name = os.path.splitext(case_filename)[0]
        
        # Load image and ground truth data
        img_filepath = os.path.join(self.img_dir, case_filename)
        gt_filepath = os.path.join(self.gt_dir, case_filename)
        
        data = np.load(img_filepath, allow_pickle=True)
        gt_data = np.load(gt_filepath)
        
        image = data["imgs"]  # Shape: [Z, Y, X]
        spacing = data["spacing"]
        gts = gt_data["gts"]  # Shape: [Z, Y, X]
        boxes = data.get("boxes", None)  # Bounding boxes if available
        
        # Normalize the image
        image = self._normalize_image(image)
        
        # Get unique classes in the ground truth
        unique_classes = np.unique(gts)
        unique_classes = unique_classes[unique_classes > 0]  # Exclude background
        num_classes = len(unique_classes)
        
        # If no bounding boxes are provided, generate them from ground truth
        if boxes is None:
            boxes = []
            for class_id in unique_classes:
                # Find where this class exists in the mask
                class_mask = (gts == class_id)
                
                # Get bounding box coordinates
                z_indices, y_indices, x_indices = np.where(class_mask)
                
                if len(z_indices) == 0:
                    continue  # Skip if class not present
                    
                z_min, z_max = np.min(z_indices), np.max(z_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                
                # Create bbox dictionary
                bbox = {
                    'class_id': int(class_id),
                    'z_min': z_min,
                    'z_max': z_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'x_min': x_min,
                    'x_max': x_max,
                    'z_mid': (z_min + z_max) // 2  # Middle slice for 2D option
                }
                
                boxes.append(bbox)
        

        # Create input tensor with image and bounding box channels
        # For simplicity, we'll use 1 channel for image and 1 channel for each bbox
        # Shape: [1+num_classes, Z, Y, X]
        

        
        # Create bbox channels (one for each class)
        bbox_channels = []
        for box in boxes:
            bbox_channel = self._create_bbox_channel(image.shape, box)
            bbox_channels.append(bbox_channel)
        
        # If there are fewer classes than expected, pad with zeros
        while len(bbox_channels) < 7:  # Assuming maximum 6 classes as in your metrics
            bbox_channels.append(np.zeros_like(image))

        image, gts, bbox_channels = crop_to_fixed_size(
        image, gts, bbox_channels, target_size=(192, 192, 192)  # Example target size
        )
        
        
        # Combine all channels
        input_tensor = np.concatenate([image[np.newaxis]] + [ch[np.newaxis] for ch in bbox_channels[:7]], axis=0)
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_tensor).float()
        gt_tensor = torch.from_numpy(gts).long()
        
        return {
            'input': input_tensor,  # Combined image and bbox channels
            'gt': gt_tensor,        # Ground truth segmentation
            #'case_name': case_name,
            #'boxes': boxes,
            #'spacing': spacing,
            #'num_classes': num_classes
        }




def custom_collate_fn(batch):
    """
    Custom collate function that handles variable-sized 3D data.
    Instead of batching tensors directly, it keeps them separate.
    """
    # Extract each component from the batch
    inputs = [item['input'] for item in batch]
    gts = [item['gt'] for item in batch]
    case_names = [item['case_name'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    spacings = [item['spacing'] for item in batch]
    num_classes = [item['num_classes'] for item in batch]
    shapes = [item['shape'] for item in batch]
    
    # Return a dictionary of lists instead of trying to batch tensors
    return {
        'input': inputs,           # List of input tensors
        'gt': gts,                 # List of ground truth tensors
        'case_name': case_names,
        'boxes': boxes,
        'spacing': spacings,
        'num_classes': num_classes,
        'shape': shapes
    }



def crop_to_fixed_size(image, mask, bbox_channels, target_size=(128, 128, 128)):
    """
    Crop volumes to a fixed size centered around the region of interest.
    
    Args:
        image: The image volume (Z, Y, X)
        mask: The ground truth mask (Z, Y, X)
        bbox_channels: List of bounding box channels
        target_size: Desired output size (Z, Y, X)
        
    Returns:
        Cropped image, mask, and bbox channels
    """
    # Find center of non-zero regions (ROI) in the mask
    if np.any(mask > 0):
        z_indices, y_indices, x_indices = np.where(mask > 0)
        center_z = (np.min(z_indices) + np.max(z_indices)) // 2
        center_y = (np.min(y_indices) + np.max(y_indices)) // 2
        center_x = (np.min(x_indices) + np.max(x_indices)) // 2
    else:
        # If mask is empty, use center of image
        center_z = image.shape[0] // 2
        center_y = image.shape[1] // 2
        center_x = image.shape[2] // 2
    
    # Calculate crop boundaries
    half_z, half_y, half_x = [size // 2 for size in target_size]
    
    # Calculate start and end indices for cropping
    z_start = max(0, center_z - half_z)
    y_start = max(0, center_y - half_y)
    x_start = max(0, center_x - half_x)
    
    z_end = min(image.shape[0], z_start + target_size[0])
    y_end = min(image.shape[1], y_start + target_size[1])
    x_end = min(image.shape[2], x_start + target_size[2])
    
    # Adjust start indices if the end boundary is limited by image size
    z_start = max(0, z_end - target_size[0])
    y_start = max(0, y_end - target_size[1])
    x_start = max(0, x_end - target_size[2])
    
    # Crop image and mask
    cropped_image = image[z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Crop bbox channels
    cropped_bbox_channels = []
    for channel in bbox_channels:
        cropped_channel = channel[z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_bbox_channels.append(cropped_channel)
    
    # Pad if necessary to reach target size
    if cropped_image.shape != target_size:
        pad_z = max(0, target_size[0] - cropped_image.shape[0])
        pad_y = max(0, target_size[1] - cropped_image.shape[1])
        pad_x = max(0, target_size[2] - cropped_image.shape[2])
        
        # Apply padding to reach target size
        pad_width = ((0, pad_z), (0, pad_y), (0, pad_x))
        cropped_image = np.pad(cropped_image, pad_width, mode='constant')
        cropped_mask = np.pad(cropped_mask, pad_width, mode='constant')
        
        for i in range(len(cropped_bbox_channels)):
            cropped_bbox_channels[i] = np.pad(cropped_bbox_channels[i], pad_width, mode='constant')
    
    return cropped_image, cropped_mask, cropped_bbox_channels

def get_dataloader(img_dir, gt_dir, batch_size=2, num_workers=4, split='train'):
    """Create a dataloader for the interactive segmentation dataset"""
    dataset = SimpleInteractiveSegmentationDataset(
        img_dir=img_dir,
        gt_dir=gt_dir,
        use_full_volume=True,  # Use full 3D volume
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":

    from src.config import config

    img_dir = os.path.join(config["DATA_DIR"], "3D_val_npz")
    gt_dir = os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg")

    # Create dataloaders
    train_loader = get_dataloader(img_dir, gt_dir, batch_size=2, split='train')
    val_loader = get_dataloader(img_dir, gt_dir, batch_size=1, split='val')

    # Example training loop
    for batch in train_loader:
        inputs = batch['input']      # Shape: [B, 1+6, Z, Y, X] 
                                    # (1 image channel + up to 6 bbox channels)
        targets = batch['gt']        # Shape: [B, Z, Y, X]
        
        # First channel is image, rest are bbox channels
        image = inputs[:, 0:1]       # Extract just the image
        bbox_channels = inputs[:, 1:] # Extract bbox channels
        
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        