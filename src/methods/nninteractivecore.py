

import numpy as np
import torch
from torch.nn.functional import interpolate
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd

import nnInteractive
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub


class nnInteractiveCorePredictor:
    """
    Simplified predictor class for nnInteractive that uses core model functionality
    with minimal preprocessing and single patch prediction.
    """
    
    def __init__(self, checkpoint_path, device=None, verbose=False):
        """
        Initialize the predictor with a trained model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the trained model folder
            device (torch.device, optional): Device to run inference on. Defaults to CUDA if available.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.verbose = verbose
        self.network = None
        self.configuration_manager = None
        self.label_manager = None
        
        # Initialize model from checkpoint
        self.initialize_from_trained_model_folder(checkpoint_path)
        
        self.log(f"Model initialized on {self.device}")
    
    def log(self, message):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")
    
    def initialize_from_trained_model_folder(self, model_training_output_dir):
        """
        Load model from a trained model folder.
        
        Args:
            model_training_output_dir (str): Path to the trained model folder
        """
        # Load model settings
        try:
            expected_json_file = join(model_training_output_dir, 'inference_session_class.json')
            json_content = load_json(expected_json_file)
            
            # Set default interaction parameters
            self.interaction_radius = 4
            if isinstance(json_content, dict) and 'point_radius' in json_content:
                self.interaction_radius = json_content['point_radius']
                
            # Load dataset and plans
            dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
            plans = load_json(join(model_training_output_dir, 'plans.json'))
            plans_manager = PlansManager(plans)
            
            # Find fold folder
            import os
            fold_folders = [f for f in os.listdir(model_training_output_dir) if f.startswith('fold_')]
            assert len(fold_folders) > 0, "No fold folders found"
            fold_folder = fold_folders[0]
            
            # Load checkpoint
            checkpoint = torch.load(
                join(model_training_output_dir, fold_folder, 'checkpoint_final.pth'),
                map_location=self.device, 
                weights_only=False
            )
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            parameters = checkpoint['network_weights']
            
            # Create configuration manager
            configuration_manager = plans_manager.get_configuration(configuration_name)
            
            # Build network
            num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
            trainer_class = recursive_find_python_class(
                join(nnInteractive.__path__[0], "trainer"),
                trainer_name, 
                'nnInteractive.trainer'
            )
            if trainer_class is None:
                self.log('Using default nnInteractiveTrainer_stub')
                trainer_class = nnInteractiveTrainer_stub
                
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                enable_deep_supervision=False
            ).to(self.device)
            network.load_state_dict(parameters)
            
            # Save necessary attributes
            self.plans_manager = plans_manager
            self.configuration_manager = configuration_manager
            self.network = network
            self.dataset_json = dataset_json
            self.label_manager = plans_manager.get_label_manager(dataset_json)
            
            self.log("Model loaded successfully")
            
        except Exception as e:
            self.log(f"Error initializing model: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess the input image for the model.
        
        Args:
            image (np.ndarray): Input image of shape [c, x, y, z]
            
        Returns:
            tuple: Preprocessed image tensor and preprocessing properties
        """
        self.log(f"Preprocessing image of shape {image.shape}")
        
        # Convert to tensor
        image_torch = torch.from_numpy(image).float()
        
        # Crop to nonzero region for efficiency
        nonzero_idx = torch.where(image_torch != 0)
        bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
        slicer = bounding_box_to_slice(bbox)
        image_torch = image_torch[slicer]
        
        # Normalize
        image_torch = (image_torch - image_torch.mean()) / image_torch.std()
        
        # Store preprocessing properties
        preproc_props = {'bbox_used_for_cropping': bbox}
        
        return image_torch, preproc_props
    
    def create_interaction_tensor(self, image_shape, coords, include_interaction):
        """
        Create a simple interaction tensor with a single point or region.
        
        Args:
            image_shape (tuple): Shape of the image
            coords (tuple): Coordinates for interaction
            include_interaction (bool): Whether to mark as foreground (True) or background (False)
            
        Returns:
            torch.Tensor: Interaction tensor
        """
        # Create a 7-channel interaction tensor (initial seg, bbox+, bbox-, point+, point-, scribble+, scribble-)
        interactions = torch.zeros((7, *image_shape[1:]), dtype=torch.float32)
        
        # If coordinates are for a bounding box
        if isinstance(coords[0], list):
            channel = 1 if include_interaction else 2  # bbox+ or bbox-
            slicer = tuple([slice(*i) for i in coords])
            interactions[(channel, *slicer)] = 1
        else:
            # For point interaction, create a simple point
            channel = 3 if include_interaction else 4  # point+ or point-
            
            # Create a small sphere around the point
            radius = self.interaction_radius
            z, y, x = coords  # Assuming ZYX order for coordinates
            
            # Get dimensions
            d, h, w = image_shape[1:]  # Assuming CZYX order for image
            
            # Create indices
            z_indices, y_indices, x_indices = torch.meshgrid(
                torch.arange(max(0, z-radius), min(d, z+radius+1)),
                torch.arange(max(0, y-radius), min(h, y+radius+1)),
                torch.arange(max(0, x-radius), min(w, x+radius+1)),
                indexing='ij'
            )
            
            # Calculate distances
            distances = torch.sqrt((z_indices - z)**2 + (y_indices - y)**2 + (x_indices - x)**2)
            
            # Create mask
            mask = distances <= radius
            
            # Set interaction
            interactions[channel, z_indices[mask], y_indices[mask], x_indices[mask]] = 1
            
        return interactions
    
    def get_patch_around_point(self, image, interactions, coords, patch_size):
        """
        Extract a patch centered around the given coordinates.
        
        Args:
            image (torch.Tensor): Preprocessed image tensor of shape [C, Z, Y, X]
            interactions (torch.Tensor): Interaction tensor of shape [7, Z, Y, X]
            coords: Either coordinates [Z, Y, X] or bounding box [[Z1, Z2], [Y1, Y2], [X1, X2]]
            patch_size (tuple): Size of the patch to extract [Z, Y, X]
            
        Returns:
            tuple: Patch image, patch interactions, patch bbox
        """
        # If coords is a bounding box, use the center
        if isinstance(coords[0], list):
            center = [round((i[0] + i[1]) / 2) for i in coords]
        else:
            center = list(coords)  # Convert to list for easier manipulation
            
        # Calculate bbox for extracting patch
        bbox = []
        for dim, (c, s) in enumerate(zip(center, patch_size)):
            half_size = s // 2
            start = max(0, c - half_size)
            # For dimension access, add 1 to skip the channel dimension in image shape
            end = min(image.shape[dim+1], c + half_size + (s % 2))
            bbox.append([start, end])
            
        # Extract patches
        slicer = tuple([slice(b[0], b[1]) for b in bbox])
        patch_img = image[:, slicer[0], slicer[1], slicer[2]]
        patch_interactions = interactions[:, slicer[0], slicer[1], slicer[2]]
        
        # Resize if needed
        if not all([(b[1]-b[0]) == s for b, s in zip(bbox, patch_size)]):
            patch_img = interpolate(patch_img.unsqueeze(0), size=patch_size, mode='trilinear').squeeze(0)
            patch_interactions = interpolate(patch_interactions.unsqueeze(0), size=patch_size, mode='nearest').squeeze(0)
            
        return patch_img, patch_interactions, bbox
    
    def predict_single_class(self, image, bbox_or_click, include_interaction=True):
        """
        Predict segmentation for a single class based on bbox or click.
        
        Args:
            image (np.ndarray): Input image of shape [c, z, y, x]
            bbox_or_click: Either bbox coordinates [[z1,z2],[y1,y2],[x1,x2]] or click coordinates [z,y,x]
            include_interaction (bool): Whether this is a foreground or background interaction
            
        Returns:
            np.ndarray: Binary segmentation result
        """
        # Preprocess image
        preprocessed_image, preproc_props = self.preprocess_image(image)
        
        # Get input shape and patch size
        input_shape = preprocessed_image.shape
        patch_size = self.configuration_manager.patch_size
        
        # Transform coordinates to preprocessed image space
        if isinstance(bbox_or_click[0], list):  # It's a bbox
            coords = []
            for i, (start, end) in enumerate(bbox_or_click):
                crop_start = preproc_props['bbox_used_for_cropping'][i+1][0]  # +1 to skip channel dim
                coords.append([start - crop_start, end - crop_start])
        else:  # It's a click
            coords = []
            for i, coord in enumerate(bbox_or_click):
                crop_start = preproc_props['bbox_used_for_cropping'][i+1][0]  # +1 to skip channel dim
                coords.append(coord - crop_start)
        
        # Create interaction tensor
        interactions = self.create_interaction_tensor(input_shape, coords, include_interaction)
        
        # Extract patch around interaction
        patch_img, patch_interactions, bbox = self.get_patch_around_point(
            preprocessed_image, interactions, coords, patch_size
        )
        
        # Move tensors to device
        patch_img = patch_img.to(self.device)
        patch_interactions = patch_interactions.to(self.device)
        
        # Create input for model
        model_input = torch.cat([patch_img, patch_interactions], dim=0).unsqueeze(0)
        
        # Run prediction
        self.network.eval()
        with torch.no_grad():
            prediction = self.network(model_input)[0].argmax(0)
        
        # Create output tensor
        result = torch.zeros(input_shape[1:], dtype=torch.uint8)
        
        # Place prediction in result
        slicer = tuple([slice(b[0], b[1]) for b in bbox])
        
        # Check if prediction shape matches the expected shape
        expected_shape = tuple(b[1] - b[0] for b in bbox)
        if prediction.shape != expected_shape:
            # Resize prediction if shapes don't match
            prediction = interpolate(
                prediction.float().unsqueeze(0).unsqueeze(0), 
                size=expected_shape, 
                mode='nearest'
            )[0, 0].byte()
        
        result[slicer] = prediction.cpu()
        
        # Convert back to original image space
        orig_result = np.zeros(image.shape[1:], dtype=np.uint8)
        
        # Get the bounding box in original image (skip channel dimension)
        bbox_orig = preproc_props['bbox_used_for_cropping'][1:]
        
        # Create slicer for original image
        orig_slicer = tuple([slice(b[0], b[1]) for b in bbox_orig])
        
        # Place result in original image
        orig_result[orig_slicer] = result.numpy()
        
        # Clean up GPU memory
        empty_cache(self.device)
        
        return orig_result
    
    def predict(self, image, spacing, bboxs=None, clicks=None, prev_pred=None, num_classes_max=None):
        """
        Predict segmentation for multiple classes.
        
        Args:
            image (np.ndarray): Input 3D image with shape [C, Z, Y, X]
            bboxs (list, optional): List of bounding boxes for prediction
            clicks (list, optional): List of clicks for prediction
            prev_pred (np.ndarray, optional): Previous prediction
            num_classes_max (int, optional): Limit number of classes processed
            
        Returns:
            tuple: Final segmentation result, inference time
        """
        import time
        start_time = time.time()
        
        # Check inputs
        if image.ndim == 3:
            # Add channel dimension if missing
            image = image[np.newaxis]
            
        self.log(f"Input image shape: {image.shape}")
        
        # Initialize output
        final_segmentation = np.zeros_like(image[0], dtype=np.uint8)
        
        # Determine mode and number of classes
        is_bbox_mode = bboxs is not None
        if is_bbox_mode:
            num_classes_prompted = len(bboxs)
            self.log(f"Mode: BBox, Classes Prompted: {num_classes_prompted}")
        elif clicks is not None:
            num_classes_prompted = len(clicks)
            self.log(f"Mode: Clicks, Classes Prompted: {num_classes_prompted}")
        else:
            self.log("No prompts provided. Returning previous prediction.")
            return prev_pred.copy() if prev_pred is not None else final_segmentation, 0
        
        # Limit classes if requested
        num_classes = num_classes_prompted
        if num_classes_max is not None:
            num_classes = min(num_classes_max, num_classes_prompted)
        
        # Process each class
        for idx in range(num_classes):
            class_id = idx + 1  # 0 is background
            self.log(f"Processing Class {class_id}")
            
            if is_bbox_mode:
                # Process bbox
                bbox = bboxs[idx]
                
                # Convert bbox format to match expected input [Z, Y, X]
                bbox_coords = [
                    [bbox["z_min"], bbox["z_max"]],
                    [bbox["z_mid_y_min"], bbox["z_mid_y_max"]],
                    [bbox["z_mid_x_min"], bbox["z_mid_x_max"]],
                ]
                self.log(f"BBox coordinates: {bbox_coords}")
                
                # Get binary prediction
                binary_pred = self.predict_single_class(image, bbox_coords, include_interaction=True)
                
            else:
                # Process clicks
                fg_clicks = clicks[idx].get("fg", [])
                bg_clicks = clicks[idx].get("bg", [])
                
                if not fg_clicks:
                    self.log(f"Skipping class {class_id}: No foreground clicks")
                    continue
                
                # Use last foreground click
                click_coords = fg_clicks[-1]  # Assuming format [Z, Y, X]
                self.log(f"Using click at {click_coords}")
                
                # Get binary prediction
                binary_pred = self.predict_single_class(image, click_coords, include_interaction=True)
            
            # Add to final segmentation
            final_segmentation[binary_pred > 0] = class_id
        
        inference_time = time.time() - start_time
        self.log(f"Prediction completed in {inference_time:.3f} seconds")
        
        return final_segmentation, inference_time


