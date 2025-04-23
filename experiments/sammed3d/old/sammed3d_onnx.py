# sam_med3d_onnx_predictor.py

import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchio as tio


def create_gt_arr(shape, point, category_index, square_size=5):
    # (Same implementation as provided)
    gt_array = np.zeros(shape)
    z, y, x = point
    half_size = square_size // 2
    z_min = max(int(z - half_size), 0)
    z_max = min(int(z + half_size) + 1, shape[0])
    y_min = max(int(y - half_size), 0)
    y_max = min(int(y + half_size) + 1, shape[1])
    x_min = max(int(x - half_size), 0)
    x_max = min(int(x + half_size) + 1, shape[2])
    gt_array[z_min:z_max, y_min:y_max, x_min:x_max] = category_index
    return gt_array


def transform_coords_to_roi(
    orig_coords, orig_spacing, target_spacing, cropping_params, padding_params
):
    """
    Transform coordinates from original image space to ROI space.

    Args:
        orig_coords: Tuple of (z, y, x) coordinates in original image space
        orig_spacing: Original voxel spacing (z, y, x)
        target_spacing: Target voxel spacing (z, y, x)
        cropping_params: Cropping parameters [z, y, x]
        padding_params: Padding parameters [z0, z1, y0, y1, x0, x1]

    Returns:
        Tuple of (z, y, x) coordinates in ROI space
    """
    # Step 1: Transform from original to resampled space
    resampled_coords = [
        orig_coords[0] * (orig_spacing[0] / target_spacing[0]),
        orig_coords[1] * (orig_spacing[1] / target_spacing[1]),
        orig_coords[2] * (orig_spacing[2] / target_spacing[2]),
    ]

    # Step 2: Transform from resampled space to ROI space
    roi_z = resampled_coords[0] - cropping_params[0] + padding_params[0]
    roi_y = resampled_coords[1] - cropping_params[2] + padding_params[2]
    roi_x = resampled_coords[2] - cropping_params[4] + padding_params[4]

    return (round(roi_z), round(roi_y), round(roi_x))


def transform_coords_from_roi(
    roi_coords, orig_spacing, target_spacing, cropping_params, padding_params
):
    """
    Transform coordinates from ROI space back to original image space.

    Args:
        roi_coords: Tuple of (z, y, x) coordinates in ROI space
        orig_spacing: Original voxel spacing (z, y, x)
        target_spacing: Target voxel spacing (z, y, x)
        cropping_params: Cropping parameters [z, y, x]
        padding_params: Padding parameters [z0, z1, y0, y1, x0, x1]

    Returns:
        Tuple of (z, y, x) coordinates in original image space
    """
    # Step 1: Transform from ROI space to resampled space
    resampled_z = roi_coords[0] - padding_params[0] + cropping_params[0]
    resampled_y = roi_coords[1] - padding_params[2] + cropping_params[2]
    resampled_x = roi_coords[2] - padding_params[4] + cropping_params[4]

    # Step 2: Transform from resampled space to original space
    orig_z = resampled_z * (target_spacing[0] / orig_spacing[0])
    orig_y = resampled_y * (target_spacing[1] / orig_spacing[1])
    orig_x = resampled_x * (target_spacing[2] / orig_spacing[2])

    return (round(orig_z), round(orig_y), round(orig_x))


# --- Predictor Class ---


class SAMMed3DPredictorONNX:
    """
    Predictor class using the ONNX SAM-Med3D model and workflow.
    """

    def __init__(self, onnx_model_path):

        print(f"Initializing SAMMed3DPredictorONNX with model: {onnx_model_path}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Still useful for torch ops like interpolate
        self.ort_session = None

        # Model-specific parameters (should match the exported ONNX model)
        self.target_spacing = (1.5, 1.5, 1.5)
        self.low_res_mask_size = (32, 32, 32)
        self.model_img_input_size = (128, 128, 128)
        self.onnx_input_names = [
            "input_image",
            "point_coords",
            "point_labels",
            "input_mask",
        ]
        self.onnx_output_names = ["output_mask", "iou_predictions"]  # From script v0_2

        try:
            ort_options = ort.SessionOptions()
            # Consider specifying providers=[...] based on available hardware for optimization
            providers = ["CPUExecutionProvider"]  # Default
            if self.device.type == "cuda":
                # Check available providers and prioritize CUDA if ONNX Runtime GPU build is installed
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                elif (
                    "DmlExecutionProvider" in available_providers
                ):  # For DirectML on Windows
                    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

            print(f"   Loading ONNX model using providers: {providers}")
            self.ort_session = ort.InferenceSession(
                onnx_model_path, sess_options=ort_options, providers=providers
            )
            print("SAMMed3DPredictorONNX: ONNX model loaded.")
        except Exception as e:
            print(f"ERROR loading ONNX model from {onnx_model_path}: {e}")
            raise RuntimeError(
                f"Failed to load ONNX model from {onnx_model_path}"
            ) from e

    def _preprocess(
        self,
        orig_img,
        orig_spacing,
        points_coords_list,
        points_labels_list,
        cls_prev_seg,
    ):
        """
        Performs preprocessing steps: Resample, Crop/Pad/Normalize, Transform Coords.
        Uses the FIRST point in points_coords_list to guide CropOrPad via cls_gt.
        """
        if not points_coords_list:
            raise ValueError("Preprocessing requires at least one point coordinate.")

        # Use the first point to create the guidance mask for CropOrPad
        # Assuming ZYX order for create_gt_arr based on previous script analysis
        first_point_zyx = (
            points_coords_list[0][2],
            points_coords_list[0][1],
            points_coords_list[0][0],
        )
        int_first_point_zyx = tuple(
            int(max(0, min(orig_img.shape[d] - 1, first_point_zyx[d])))
            for d in range(3)
        )
        # cls_gt only guides cropping, category index doesn't matter much here
        cls_gt = create_gt_arr(orig_img.shape, int_first_point_zyx, category_index=1)

        # --- Resampling ---
        # NOTE: Ensure orig_spacing order matches zip expectation in Resample target calculation
        # Assuming orig_spacing is XYZ from eval framework, target_spacing is ZYX? Let's assume XYZ for both here for simplicity based on `zip` usage.
        # If target_spacing should be ZYX, adapt accordingly.
        resample_target = [
            t / o for o, t in zip(orig_spacing, self.target_spacing)
        ]  # Element-wise division

        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=orig_img[None].astype(np.float32)
            ),  # Ensure float32
            label=tio.LabelMap(tensor=cls_gt[None]),  # Crop guidance
            prev_seg=tio.LabelMap(
                tensor=cls_prev_seg[None].astype(np.uint8)
            ),  # Ensure type uint8
        )
        resampler = tio.Resample(target=resample_target)
        subject = resampler(subject)

        # --- Cropping/Padding & Normalization ---
        crop_transform = tio.CropOrPad(
            mask_name="label", target_shape=self.model_img_input_size
        )
        padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
        if cropping_params is None:
            cropping_params = (0, 0, 0, 0, 0, 0)
        if padding_params is None:
            padding_params = (0, 0, 0, 0, 0, 0)

        infer_transform = tio.Compose(
            [
                crop_transform,
                tio.ZNormalization(
                    masking_method=lambda x: x > 0
                ),  # Normalize after cropping
            ]
        )
        subject_roi = infer_transform(subject)
        roi_img = subject_roi.image.data  # Shape [1, D, H, W] (torchio uses DHW)

        # Metadata for coordinate transform and postprocessing
        meta_info = {
            "padding_params": padding_params,
            "cropping_params": cropping_params,
            "orig_shape": orig_img.shape,
            "resampled_shape": subject.spatial_shape,
        }

        # --- Transform ALL Point Coordinates ---
        transformed_coords = []
        for p_coords_xyz in points_coords_list:
            # transform_coords_to_roi expects point in XYZ order
            t_coords_xyz = transform_coords_to_roi(
                p_coords_xyz,
                orig_spacing,
                self.target_spacing,  # Pass XYZ spacings
                cropping_params,
                padding_params,
            )
            transformed_coords.append(t_coords_xyz)

        # Convert to ONNX input format: numpy arrays
        onnx_points_coords = np.array(
            [transformed_coords], dtype=np.float32
        )  # Shape: [1, NumPoints, 3]
        onnx_points_labels = np.array(
            [points_labels_list], dtype=np.int64
        )  # Shape: [1, NumPoints]

        # --- Prepare Low-Res Previous Mask ---
        # Need to resample/crop the full cls_prev_seg and then downsample
        roi_prev_seg_dhw = subject_roi.prev_seg.data  # Shape [1, D, H, W]
        print(roi_prev_seg_dhw.float().shape)
        
        # Interpolate to low_res_mask_size (D/4, H/4, W/4) - Assuming model_img_input_size D, H, W maps to low_res D/4, H/4, W/4
        # Note: low_res_mask_size defined as (32,32,32) matching D/4 etc.
        onnx_prev_mask = (
            F.interpolate(
                roi_prev_seg_dhw.float(),  # Ensure float for interpolate
                size=(1,self.low_res_mask_size[0], self.low_res_mask_size[1], self.low_res_mask_size[2]),  # Target size D/4, H/4, W/4
                mode="nearest",  # Use nearest for masks
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # Shape [1, 1, 32, 32, 32]

        return (
            roi_img.cpu().numpy().astype(np.float32),
            onnx_points_coords,
            onnx_points_labels,
            onnx_prev_mask,
            meta_info,
        )

    def _postprocess(self, onnx_output_mask, meta_info):
        """
        Performs postprocessing: Threshold, Uncrop/Unpad, Resample back.
        """
        # output_mask_onnx shape is [1, 1, D, H, W] corresponding to model_img_input_size
        # Thresholding (already done in model v0.2, just ensure type)
        output_mask = onnx_output_mask.squeeze().astype(np.uint8)  # Shape [D, H, W]
        if output_mask.shape != self.model_img_input_size:
            print(
                f"Warning: ONNX output mask shape {output_mask.shape} doesn't match model input size {self.model_img_input_size}"
            )
            # Handle potential mismatch if necessary

        # Map back from ROI to Resampled space
        pred3D_full_resampled = np.zeros(meta_info["resampled_shape"], dtype=np.uint8)
        padding_params = meta_info["padding_params"]
        cropping_params = meta_info["cropping_params"]

        # Calculate slices for unpadding and target region based on padding/cropping
        # Be careful with torchio padding/cropping format (W_padd_left, W_padd_right, H_padd_left, ...)
        # Assuming D, H, W order for model_img_input_size and padding_params
        D, H, W = self.model_img_input_size
        D_pad_pre, D_pad_post, H_pad_pre, H_pad_post, W_pad_pre, W_pad_post = (
            padding_params
        )
        D_crop_pre, _, H_crop_pre, _, W_crop_pre, _ = (
            cropping_params  # We only need the start crop offset
        )

        # Source slices from the ROI output mask (removing padding)
        src_D_slice = slice(D_pad_pre, D - D_pad_post)
        src_H_slice = slice(H_pad_pre, H - H_pad_post)
        src_W_slice = slice(W_pad_pre, W - W_pad_post)

        # Target slices in the full resampled array (applying crop offset)
        # Target size is the unpadded size
        target_D_size = D - D_pad_pre - D_pad_post
        target_H_size = H - H_pad_pre - H_pad_post
        target_W_size = W - W_pad_pre - W_pad_post

        target_D_slice = slice(D_crop_pre, D_crop_pre + target_D_size)
        target_H_slice = slice(H_crop_pre, H_crop_pre + target_H_size)
        target_W_slice = slice(W_crop_pre, W_crop_pre + target_W_size)

        try:
            pred3D_full_resampled[target_D_slice, target_H_slice, target_W_slice] = (
                output_mask[src_D_slice, src_H_slice, src_W_slice]
            )
        except ValueError as e:
            print("Error during uncropping/unpadding - shape mismatch likely:")
            print(f"  Target shape: {pred3D_full_resampled.shape}")
            print(
                f"  Target slices: D{target_D_slice}, H{target_H_slice}, W{target_W_slice}"
            )
            print(f"  Source shape: {output_mask.shape}")
            print(f"  Source slices: D{src_D_slice}, H{src_H_slice}, W{src_W_slice}")
            raise e

        # Resample back to Original image space
        pred3D_full_ori = (
            F.interpolate(
                torch.from_numpy(pred3D_full_resampled)[
                    None, None
                ].float(),  # Add Batch, Channel, ensure float
                size=meta_info["orig_shape"],  # Target original shape (Z, Y, X) ?
                mode="nearest",  # Use nearest for masks
            )
            .cpu()
            .numpy()
            .squeeze()
            .astype(np.uint8)
        )

        return pred3D_full_ori

    def predict(
        self,
        image_data,  # Full 3D image (NumPy ZYX)
        spacing_data,  # Voxel spacing (tuple/list XYZ)
        bbox_data=None,  # BBox list (iter 0) [{'z_min'.., 'z_mid_x_min'..}, ...]
        clicks_data=None,  # Click list (iter 1+) [{'fg':[[x,y,z],..], 'bg':[[x,y,z],..]}, ...]
        prev_pred_data=None,  # Full prediction from previous step (NumPy ZYX)
        num_classes_max=None # Max number of classes to predict (optional)
    ):
        """
        Performs SAM-Med3D ONNX inference using the defined workflow.
        """
        start_time = time.time()
        print("--- SAMMed3DPredictorONNX.predict() start ---")

        if self.ort_session is None:
            print("   ERROR: ONNX session not loaded.")
            return np.zeros_like(image_data, dtype=np.uint8), time.time() - start_time

        final_segmentation = np.zeros_like(image_data, dtype=np.uint8)
        if prev_pred_data is None:
            prev_pred_data = np.zeros_like(image_data, dtype=np.uint8)

        num_classes_prompted = 0
        is_bbox_iteration = False
        if bbox_data is not None:
            num_classes_prompted = len(bbox_data)
            is_bbox_iteration = True
        elif clicks_data is not None:
            num_classes_prompted = len(clicks_data)
        else:
            return prev_pred_data.copy(), time.time() - start_time  # No prompts
        
        num_classes = num_classes_prompted if num_classes_max is None else min(num_classes_prompted, num_classes_max)

        # --- Iterate through classes ---
        for idx in range(num_classes):
            class_index = idx + 1
            print(f"--- Processing Class Index: {class_index} ---")

            cls_prev_seg_mask = prev_pred_data == class_index  # Boolean mask ZYX
            # Copy previous prediction first
            final_segmentation[cls_prev_seg_mask] = class_index

            # --- Collect ALL points and labels for this class/iteration ---
            # Unlike the previous version, ONNX likely takes all points at once
            points_coords_list_xyz = []
            points_labels_list = []

            if is_bbox_iteration:
                bbox = bbox_data[idx]
                # Bbox is always positive label = 1
                # Calculate center point in XYZ order
                x_cen = (bbox["z_mid_x_min"] + bbox["z_mid_x_max"]) / 2
                y_cen = (bbox["z_mid_y_min"] + bbox["z_mid_y_max"]) / 2
                z_cen = (bbox["z_min"] + bbox["z_max"]) / 2  # Assuming Z is first dim
                points_coords_list_xyz.append((x_cen, y_cen, z_cen))
                points_labels_list.append(1)
                print(
                    f"   Using BBox center as prompt point (XYZ): {points_coords_list_xyz[-1]}"
                )

            else:  # Click iteration
                # Add all FG clicks (label 1)
                for click_xyz in clicks_data[idx].get("fg", []):
                    points_coords_list_xyz.append(click_xyz)
                    points_labels_list.append(1)
                # Add all BG clicks (label 0 - CHECK if ONNX expects 0 or -1)
                # Assuming label 0 for background based on typical SAM usage
                for click_xyz in clicks_data[idx].get("bg", []):
                    points_coords_list_xyz.append(click_xyz)
                    points_labels_list.append(0)  # Use 0 for background

                if not points_coords_list_xyz:
                    print(f"   Skipping class {class_index}: No clicks provided.")
                    continue
                print(
                    f"   Using {len(points_coords_list_xyz)} click(s) as prompt points."
                )

            try:
                # 1. Preprocess: Resample, Crop/Pad/Norm, Transform points
                # Pass image_data (ZYX), spacing (XYZ), points (list of XYZ), prev_seg (ZYX)
                (
                    roi_img_np,
                    points_coords_np,
                    points_labels_np,
                    prev_mask_np,
                    meta_info,
                ) = self._preprocess(
                    image_data,
                    spacing_data,
                    points_coords_list_xyz,
                    points_labels_list,
                    cls_prev_seg_mask,
                )
                # roi_img_np shape: [1, D, H, W]

                # 2. Prepare ONNX Inputs
                onnx_inputs = {
                    "input_image": roi_img_np,
                    "point_coords": points_coords_np,  # Shape [1, NumPoints, 3]
                    "point_labels": points_labels_np,  # Shape [1, NumPoints], MUST be int64
                    "input_mask": prev_mask_np,  # Shape [1, 1, 32, 32, 32]
                }
                # Verify dtypes
                if onnx_inputs["point_labels"].dtype != np.int64:
                    print(
                        f"Warning: Casting point_labels dtype from {onnx_inputs['point_labels'].dtype} to int64."
                    )
                    onnx_inputs["point_labels"] = onnx_inputs["point_labels"].astype(
                        np.int64
                    )

                print(f"   Preprocessing done. ROI shape: {roi_img_np.shape}")
                print("   ONNX Inputs:")
                print(
                    f"     input_image: {onnx_inputs['input_image'].shape}, {onnx_inputs['input_image'].dtype}"
                )
                print(
                    f"     point_coords: {onnx_inputs['point_coords'].shape}, {onnx_inputs['point_coords'].dtype}"
                )
                print(
                    f"     point_labels: {onnx_inputs['point_labels'].shape}, {onnx_inputs['point_labels'].dtype}"
                )
                print(
                    f"     input_mask: {onnx_inputs['input_mask'].shape}, {onnx_inputs['input_mask'].dtype}"
                )

                # 3. Run ONNX Inference
                print("   Running ONNX inference...")
                output_mask_onnx, _ = (
                    self.ort_session.run(  # Ignore iou_predictions for now
                        self.onnx_output_names, onnx_inputs
                    )
                )
                # output_mask_onnx shape [1, 1, D, H, W] - matches model_img_input_size
                print(
                    f"   Inference done. Raw ONNX Output Mask Shape: {output_mask_onnx.shape}"
                )

                # 4. Postprocess
                if output_mask_onnx is not None and np.any(
                    output_mask_onnx > 0.5
                ):  # Check if output is valid and potentially non-zero
                    print("   Postprocessing...")
                    pred_ori = self._postprocess(output_mask_onnx, meta_info)
                    print(f"   Postprocessing done. Final Mask Shape: {pred_ori.shape}")

                    # 5. Aggregate Result
                    update_mask = pred_ori != 0
                    final_segmentation[update_mask] = class_index
                    print(
                        f"   Updated final segmentation for class {class_index}. New non-zero voxels: {np.sum(update_mask)}"
                    )
                else:
                    print(
                        "   Skipping postprocessing: ONNX output mask is empty or invalid."
                    )

            except Exception as e:
                print(f"   ERROR during ONNX steps for class {class_index}: {e}")
                import traceback

                traceback.print_exc()
                print(
                    f"   Keeping previous prediction for class {class_index} due to error."
                )

        # --- End of class loop ---
        inference_time = time.time() - start_time
        print(
            f"--- SAMMed3DPredictorONNX.predict() end. Time: {inference_time:.2f}s ---"
        )

        return final_segmentation, inference_time
