from typing import Union, List, Tuple
import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from typing import Union, List, Tuple
import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from monai.transforms import (
    RandSpatialCropd,
    RandAffined,
    Compose,
    SpatialPadd,
    CenterSpatialCropd,
)


class NormalizeSingleImageTransform(AbstractTransform):
    def __call__(self, **data_dict):
        imgs = data_dict["image"]
        assert imgs.shape[0] == 1
        means = torch.mean(imgs, dim=list(range(1, imgs.ndim)), keepdim=True)
        stds = torch.std(imgs, dim=list(range(1, imgs.ndim)), keepdim=True)
        eps = 1e-6
        normed_imgs = (imgs - means) / (stds + eps)
        data_dict["image"] = normed_imgs
        return data_dict


def mask2D_to_bbox(
    gt2D,
):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, 6, 1)[0]
    scale_y, scale_x = gt2D.shape
    bbox_shift_x = int(bbox_shift * scale_x / 256)
    bbox_shift_y = int(bbox_shift * scale_y / 256)
    # print(f'{bbox_shift_x=} {bbox_shift_y=} with orig {bbox_shift=}')
    x_min = max(0, x_min - bbox_shift_x)
    x_max = min(W - 1, x_max + bbox_shift_x)
    y_min = max(0, y_min - bbox_shift_y)
    y_max = min(H - 1, y_max + bbox_shift_y)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes


def mask3D_to_bbox(
    gt3D,
):
    b_dict = {}
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    z_indices = np.unique(z_indices)
    # middle of z_indices
    z_middle = z_indices[len(z_indices) // 2]

    D, H, W = gt3D.shape
    b_dict["z_min"] = z_min
    b_dict["z_max"] = z_max
    b_dict["z_mid"] = z_middle

    gt_mid = gt3D[z_middle]

    box_2d = mask2D_to_bbox(
        gt_mid,
    )
    x_min, y_min, x_max, y_max = box_2d
    b_dict["z_mid_x_min"] = x_min
    b_dict["z_mid_y_min"] = y_min
    b_dict["z_mid_x_max"] = x_max
    b_dict["z_mid_y_max"] = y_max

    assert z_min == max(0, z_min)
    assert z_max == min(D - 1, z_max)
    return b_dict


class AddBBoxAndEmptyChannelsSingleClassTransform(AbstractTransform):
    def __init__(self, only_2d_bbox, class_idx=None):
        """
        Takes the input data and adds 7 prompt channels to it, fills bbox into channel 1/-6:
        - Channel 0: Previous segmentation
        - Channel -6: Positive bounding box/lasso
        - Channel -5: Negative bounding box/lasso
        - Channel -4: Positive point interaction
        - Channel -3: Negative point interaction
        - Channel -2: Positive scribble
        - Channel -1: Negative scribble

        """
        print("Initializing AddBBoxAndEmptyChannelsSingleClassTransform")
        self.only_2d_bbox = only_2d_bbox
        self.class_idx = class_idx

    def __call__(self, **data_dict):
        # Get the input image and ground truth
        imgs = data_dict["image"]
        gts = data_dict["segmentation"]

        # get ground truth segmentation
        # only of this image, no list, no channels
        if not (hasattr(gts, "ndim")):
            # with deep supervision, may be list
            gts = gts[0]
        assert gts.shape[0] == 1
        gts = gts[0]
        assert gts.ndim == 3

        this_coords = torch.argwhere(gts > 0)
        prompt_channels = torch.zeros((7, *imgs.shape[1:]), device=imgs.device)
        if len(this_coords) > 0:
            # otherwise no ground truth class there, so also no bbox...
            i_starts, i_stops = (
                torch.min(this_coords, axis=0).values,
                torch.max(this_coords, axis=0).values + 1,
            )

            if self.only_2d_bbox:
                prompt_channels[
                    1,
                    (i_starts[0] + i_stops[0]) // 2,
                    i_starts[1] : i_stops[1],
                    i_starts[2] : i_stops[2],
                ] = 1
            else:
                prompt_channels[
                    1,
                    i_starts[0] : i_stops[0],
                    i_starts[1] : i_stops[1],
                    i_starts[2] : i_stops[2],
                ] = 1

        # Concatenate the original image with the prompt channels,
        data_dict["image"] = torch.cat([imgs, prompt_channels], dim=0)
        return data_dict


class AddSegToImageTransform(AbstractTransform):
    def __init__(
        self,
    ):
        """
        Add segmentation to image as last channel.

        """
        print("Initializing AddSegToImageTransform")

    def __call__(self, **data_dict):
        # also add the ground turth needed by the model wrapper to generate clicks
        # will not be seen by the model itself at all, will be removed by model wrapper before
        imgs = data_dict["image"]
        # due to deep supervision there may be multiple ground truth masks, first one is full one
        gts = data_dict["segmentation"]
        if not (hasattr(gts, "ndim") and (gts.ndim == 4)):
            gts = gts[0]
        assert gts.ndim == 4

        data_dict["image"] = torch.cat([imgs, gts], dim=0)
        return data_dict


class MONAIRandSpatialTransform:
    def __call__(self, **data_dict):
        pad_transform = SpatialPadd(
            keys=["image", "segmentation"],
            spatial_size=(192, 192, 192),
            mode="constant",  # or "edge", "reflect", etc.
            value=0,  # padding value, if using "constant"
        )
        crop_transform = RandSpatialCropd(
            keys=["image", "segmentation"],
            roi_size=[192, 192, 192],
            random_size=False,  # disables true random cropping
        )
        affine_transform = RandAffined(
            keys=["image", "segmentation"],
            prob=0.2,  # For rotation and scaling together
            rotate_range=[0.52, 0.52, 0.52],  # ~30 degrees in all axes
            scale_range=[0.3, 0.3, 0.3],  # (1 - 0.7, 1.4 - 1)
            mode=["bilinear", "nearest"],  # image, label interpolation
            padding_mode="zeros",
        )
        transform = Compose([pad_transform, crop_transform, affine_transform])

        return transform(data_dict)


class MONAIFixedSpatialTransform:
    def __call__(self, **data_dict):
        pad_transform = SpatialPadd(
            keys=["image", "segmentation"],
            spatial_size=(192, 192, 192),
            mode="constant",  # or "edge", "reflect", etc.
            value=0,  # padding value, if using "constant"
        )
        crop_transform = CenterSpatialCropd(
            keys=["image", "segmentation"],
            roi_size=[192, 192, 192],
        )
        transform = Compose(
            [
                pad_transform,
                crop_transform,
            ]
        )

        return transform(data_dict)
