from acvl_utils.cropping_and_padding.bounding_boxes import (
    bounding_box_to_slice,
    crop_and_pad_nd,
)

from torch.nn.functional import interpolate
from typing import Union, List, Tuple, Optional
import torch
import time
import numpy as np


def paste_tensor_leading_dim(target: torch.Tensor, source: torch.Tensor, bbox):
    """
    Paste a source tensor into a target tensor using a given bounding box.

    Both tensors are assumed to be 3D.
    The bounding box is specified in the coordinate system of the target as:
      [[x1, x2], [y1, y2], [z1, z2]]
    and its size is assumed to be equal to the shape of the source tensor.
    The bbox may exceed the boundaries of the target tensor.

    The function computes the valid overlapping region between the bbox and the target,
    and then adjusts the corresponding region in the source tensor so that only the valid
    parts are pasted.

    Args:
        target (torch.Tensor): The target tensor of shape (T0, T1, T2).
        source (torch.Tensor): The source tensor of shape (S0, S1, S2). It must be the same size as
                               the bbox, i.e. S0 = x2 - x1, etc.
        bbox (list or tuple): List of intervals for each dimension: [[x1, x2], [y1, y2], [z1, z2]].

    Returns:
        torch.Tensor: The target tensor after pasting in the source.
    """
    assert target.ndim == 4
    assert source.ndim == 4
    assert len(bbox) == 3
    target_shape = target.shape[1:]  # (T0, T1, T2)

    # For each dimension compute:
    #   - The valid region in the target: [t_start, t_end)
    #   - The corresponding region in the source: [s_start, s_end)
    target_indices = []
    source_indices = []

    for i, (b0, b1) in enumerate(bbox):
        # Determine valid region in target tensor:
        t_start = max(b0, 0)
        t_end = min(b1, target_shape[i])
        # If there's no overlap in any dimension, nothing gets pasted.
        if t_start >= t_end:
            return target

        # Determine corresponding indices in the source tensor.
        # The source's coordinate 0 corresponds to b0 in the target.
        s_start = t_start - b0
        s_end = s_start + (t_end - t_start)

        target_indices.append((t_start, t_end))
        source_indices.append((s_start, s_end))

    target[
        :,
        target_indices[0][0] : target_indices[0][1],
        target_indices[1][0] : target_indices[1][1],
        target_indices[2][0] : target_indices[2][1],
    ] = source[
        :,
        source_indices[0][0] : source_indices[0][1],
        source_indices[1][0] : source_indices[1][1],
        source_indices[2][0] : source_indices[2][1],
    ]

    return target


def preproc_image(image):
    # Convert and clone the image tensor.
    image_torch = torch.clone(torch.from_numpy(image))

    nonzero_idx = torch.where(image_torch != 0)
    # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices.
    bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
    del nonzero_idx
    slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
    image_torch = image_torch[slicer].float()

    # Normalize the cropped image.
    image_torch -= image_torch.mean()
    image_torch /= image_torch.std()
    preprocessed_props = {"bbox_used_for_cropping": bbox[1:]}
    return image_torch, preprocessed_props


def add_bbox_interaction(
    bbox_coords,
    interactions,
    preprocessed_props,
    image_shape,
    include_interaction: bool,
    only_2d_bbox: bool,
) -> np.ndarray:
    lbs_transformed = [
        round(i)
        for i in transform_coordinates_noresampling(
            [i[0] for i in bbox_coords], preprocessed_props["bbox_used_for_cropping"]
        )
    ]
    ubs_transformed = [
        round(i)
        for i in transform_coordinates_noresampling(
            [i[1] for i in bbox_coords], preprocessed_props["bbox_used_for_cropping"]
        )
    ]
    transformed_bbox_coordinates = [
        [i, j] for i, j in zip(lbs_transformed, ubs_transformed)
    ]

    for dim in range(len(transformed_bbox_coordinates)):
        transformed_start, transformed_end = transformed_bbox_coordinates[dim]

        # Clip to image boundaries
        transformed_start = max(0, transformed_start)
        transformed_end = min(
            image_shape[dim + 1], transformed_end
        )  # +1 to skip channel dim

        # Ensure the bounding box does not collapse to a single point
        if transformed_end <= transformed_start:
            if transformed_start == 0:
                transformed_end = min(1, image_shape[dim + 1])
            else:
                transformed_start = max(transformed_start - 1, 0)

        transformed_bbox_coordinates[dim] = [transformed_start, transformed_end]

    # this is for 2d bbox
    if only_2d_bbox:
        z_mid = int(np.mean(transformed_bbox_coordinates[0]))
        transformed_bbox_coordinates[0][0] = z_mid
        transformed_bbox_coordinates[0][1] = z_mid + 1
    # place bbox
    slicer = tuple([slice(*i) for i in transformed_bbox_coordinates])
    channel = -6 if include_interaction else -5
    interactions[(channel, *slicer)] = 1
    return interactions, transformed_bbox_coordinates


def transform_coordinates_noresampling(
    coords_orig: Union[List[int], Tuple[int, ...]],
    nnunet_preprocessing_crop_bbox: List[Tuple[int, int]],
) -> Tuple[int, ...]:
    """
    converts coordinates in the original uncropped image to the internal cropped representation. Man I really hate
    nnU-Net's crop to nonzero!
    """
    return tuple(
        [
            coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0]
            for d in range(len(coords_orig))
        ]
    )


def pred_all_classes(network, image, boxes, do_autozoom, only_2d_bbox_interaction):
    patch_size = (192, 192, 192)
    use_pinned_memory = False
    device = next(network.parameters()).device
    all_preds = []
    for this_bbox in boxes:
        bbox_tuple = [
            [this_bbox["z_min"], this_bbox["z_max"] + 1],
            [this_bbox["z_mid_y_min"], this_bbox["z_mid_y_max"] + 1],
            [this_bbox["z_mid_x_min"], this_bbox["z_mid_x_max"] + 1],
        ]

        target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device="cuda")
        preds_buffer = torch.full(
            (2,) + image.shape, float("nan"), dtype=torch.float32, device="cuda"
        )

        start_predict = time.time()
        preproced_image, preprocessed_props = preproc_image(image[None])  #

        interactions = torch.zeros(
            (7, *preproced_image.shape[1:]),
            device=device,
            dtype=torch.float16,
            pin_memory=(device.type == "cuda" and use_pinned_memory),
        )

        interactions, transformed_bbox = add_bbox_interaction(
            bbox_tuple,
            interactions,
            preprocessed_props,
            image_shape=preproced_image.shape,
            include_interaction=True,
            only_2d_bbox=only_2d_bbox_interaction,
        )

        bbox_center = [round((i[0] + i[1]) / 2) for i in transformed_bbox]
        bbox_size = [i[1] - i[0] for i in transformed_bbox]

        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, patch_size)]
        zoom_out_factors_from_bbox = max(
            1, max([i / j for i, j in zip(requested_size, patch_size)])
        )

        with torch.autocast("cuda", enabled=True):
            prediction_center = bbox_center
            initial_zoom_out_factor = zoom_out_factors_from_bbox

            # make a prediction at initial zoom out factor. If more zoom out is required, do this until the
            # entire object fits the FOV. Then go back to original resolution and refine.

            # we need this later.
            # previous_prediction = torch.clone(self.interactions[0])

            if not do_autozoom:
                initial_zoom_out_factor = 1

            initial_zoom_out_factor = min(initial_zoom_out_factor, 4)
            zoom_out_factor = initial_zoom_out_factor
            max_zoom_out_factor = initial_zoom_out_factor

            start_autozoom = time.time()
            max_zoom_out_factor = max(max_zoom_out_factor, zoom_out_factor)
            # initial prediction at initial_zoom_out_factor
            scaled_patch_size = [round(i * zoom_out_factor) for i in patch_size]
            scaled_bbox = [
                [c - p // 2, c + p // 2 + p % 2]
                for c, p in zip(prediction_center, scaled_patch_size)
            ]
            crop_img, pad = crop_to_valid(preproced_image, scaled_bbox)
            crop_img = crop_img.to(device, non_blocking=device.type == "cuda")
            crop_interactions, pad_interaction = crop_to_valid(interactions, scaled_bbox)

            if not all([i == j for i, j in zip(patch_size, scaled_patch_size)]):
                crop_img = interpolate(
                    pad_cropped(crop_img, pad)[None]
                    if any([x for y in pad_interaction for x in y])
                    else crop_img[None],
                    patch_size,
                    mode="trilinear",
                )[0]
            else:
                # crop_img is already on device
                crop_img = (
                    pad_cropped(crop_img, pad)
                    if any([x for y in pad_interaction for x in y])
                    else crop_img
                )

                crop_interactions = (
                    pad_cropped(crop_interactions, pad_interaction)
                    if any([x for y in pad_interaction for x in y])
                    else crop_interactions
                )

            input_for_predict = torch.cat((crop_img, crop_interactions))
            # del crop_img, crop_interactions

            # we may have to unwrap net pred in case of deep supervision
            net_pred = network(input_for_predict[None])[0]
            # now normalize deep supervision and non deep supervision case
            if net_pred.ndim == 4:
                net_pred = net_pred[None]
            assert net_pred.ndim == 5

            pred = net_pred[0].argmax(0).detach()

            # resize prediction to correct size and place in target buffer + interactions
            if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                scaled_net_probs = interpolate(
                    net_pred.to(float), scaled_patch_size, mode="trilinear"
                )[0]
            else:
                scaled_net_probs = net_pred[0]

        # place into target buffer
        bbox = [
            [i[0] + bbc[0], i[1] + bbc[0]]
            for i, bbc in zip(scaled_bbox, preprocessed_props["bbox_used_for_cropping"])
        ]
        paste_tensor(target_buffer, pred, bbox)
        paste_tensor_leading_dim(preds_buffer, scaled_net_probs, bbox)
        all_preds.append(preds_buffer)
    return all_preds
