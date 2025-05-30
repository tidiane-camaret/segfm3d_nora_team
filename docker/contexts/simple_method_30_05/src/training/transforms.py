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


class NormalizeSingleImageTransformNumpy(AbstractTransform):
    def __call__(self, **data_dict):
        img = data_dict["image"]
        assert img.ndim == 3
        eps = 1e-6
        normed_img = (img - np.mean(img)) / (np.std(img) + eps)
        data_dict["image"] = normed_img
        return data_dict


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
            # i_starts, i_stops = (
            #     torch.min(this_coords, axis=0).values,
            #     torch.max(this_coords, axis=0).values + 1,
            # )
            # use their logic again
            bbox = mask3D_to_bbox(gts > 0)

            if self.only_2d_bbox:
                prompt_channels[
                    1,
                    bbox["z_mid"],
                    bbox["z_mid_y_min"] : bbox["z_mid_y_max"],
                    bbox["z_mid_x_min"] : bbox["z_mid_x_max"],
                ] = 1
            else:
                prompt_channels[
                    1,
                    bbox["z_min"] : bbox["z_max"],
                    bbox["z_mid_y_min"] : bbox["z_mid_y_max"],
                    bbox["z_mid_x_min"] : bbox["z_mid_x_max"],
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
    def __init__(
        self,
        rotate_range=(0.52, 0.52, 0.52),
        scale_range=(0.3, 0.3, 0.3),
        prob_affine=0.2,
    ):
        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.prob_affine = prob_affine

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
            prob=self.prob_affine,  # For rotation and scaling together
            rotate_range=self.rotate_range,  # ~30 degrees in all axes
            scale_range=self.scale_range,  # (1 - 0.7, 1.4 - 1)
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


class WrapMONAI:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, **data_dict):
        return self.transform(data_dict)


def get_nonzero_bbox(image):
    nonzero_coords = np.array(np.nonzero(image))
    bbox = [
        [start, stop]
        for start, stop in zip(
            np.min(nonzero_coords, axis=1),
            np.max(nonzero_coords, axis=1) + 1,
            strict=True,
        )
    ]
    return np.array(bbox)


def bbox_to_slicer(bbox):
    slicer = [slice(start, stop) for start, stop in bbox]
    return tuple(slicer)


def get_nonzero_slicer(image):
    bbox = get_nonzero_bbox(image)
    return bbox_to_slicer(bbox)


def crop_to_nonzero_image(image, segmentation):
    slicer = get_nonzero_slicer(image)

    image = image[tuple(slicer)]
    segmentation = segmentation[tuple(slicer)]
    return image, segmentation, slicer


class CropToNonzeroImage:
    def __call__(self, image, segmentation):
        image, segmentation, _ = crop_to_nonzero_image(image, segmentation)
        return dict(image=image, segmentation=segmentation)


class ChooseClass:
    def __init__(self, fixed_class=None):
        self.fixed_class = fixed_class

    def __call__(self, image, segmentation):
        if np.max(segmentation) > 0:
            if self.fixed_class is None:
                uniq_classes = [c for c in np.unique(segmentation) if c != 0]
                wanted_class_idx = np.random.choice(uniq_classes)
            else:
                wanted_class_idx = self.fixed_class
            segmentation = segmentation == wanted_class_idx
        else:
            segmentation = segmentation.astype(bool)
        return dict(image=image, segmentation=segmentation)


class AddEmptyChan:
    def __init__(self, fixed_class=None):
        self.fixed_class = fixed_class

    def __call__(self, **data_dict):
        return {key: val[None] for key, val in data_dict.items()}


def crop_and_add_bbox(image, bbox, segmentation, patch_sizes):
    patch_sizes = np.array(patch_sizes)
    bbox_chan = np.zeros_like(image).astype(bool)
    bbox_chan[tuple(slice(i, j) for i, j in bbox)] = 1

    bbox_center = np.mean(bbox, axis=1).round()

    bbox_size = np.diff(bbox, axis=1).squeeze(1)

    bbox_size_with_margin = bbox_size + patch_sizes / 3
    # ensure the patch is large enough to capture bounding box
    # and retains aspect ratio
    zoom_out_factor = max(np.max(bbox_size_with_margin / patch_sizes), 1)
    scaled_patch_size = np.ceil(patch_sizes * zoom_out_factor).astype(np.int32)

    # compute i_starts and i_stops of cropping box
    i_starts = np.int32(bbox_center - scaled_patch_size // 2)
    i_stops = np.int32(i_starts + scaled_patch_size)

    # to keep bbox in center we may have to pad sth before and after
    # if cropping box is outside image
    n_pad_before = np.clip(-i_starts, a_min=0, a_max=None).astype(np.int32)
    n_pad_after = np.clip(i_stops - np.array(image.shape), a_min=0, a_max=None).astype(
        np.int32
    )
    assert len(n_pad_before) == len(n_pad_after)
    assert min(n_pad_before) >= 0
    assert min(n_pad_after) >= 0

    # now first crop, here ensure that you do not collapse any dim
    # scaled_orig_bbox is just computed here additionally for eval
    # this may go out of bounds of image, including negative values
    scaled_orig_bbox = np.array(
        [[i_start, i_stop] for i_start, i_stop in zip(i_starts, i_stops, strict=True)]
    )
    scaled_slicer = tuple(
        slice(max(i, 0), max(j, 1)) for i, j in zip(i_starts, i_stops, strict=True)
    )
    cropped_img = image[scaled_slicer]
    cropped_seg = segmentation[scaled_slicer]
    cropped_bbox_chan = bbox_chan[scaled_slicer]

    if (np.max(n_pad_before) > 0) or (np.max(n_pad_after) > 0):
        padded_img = np.pad(
            cropped_img,
            tuple((n_a, n_b) for n_a, n_b in zip(n_pad_before, n_pad_after)),
        )

        padded_seg = np.pad(
            cropped_seg,
            tuple((n_a, n_b) for n_a, n_b in zip(n_pad_before, n_pad_after)),
        )
        padded_bbox_chan = np.pad(
            cropped_bbox_chan,
            tuple((n_a, n_b) for n_a, n_b in zip(n_pad_before, n_pad_after)),
        )
    else:
        padded_img = cropped_img
        padded_seg = cropped_seg
        padded_bbox_chan = cropped_bbox_chan
        scaled_orig_bbox= np.array([[0,patch_sizes[0]],[0,patch_sizes[1]],[0,patch_sizes[2]]])
        scaled_slicer = tuple((slice(i,j) for i,j in scaled_orig_bbox))

    assert np.sum(padded_bbox_chan) == np.sum(bbox_chan), (
        "should be no loss of bounding box"
    )
    assert padded_img.shape == tuple(scaled_patch_size), (
        f"Expected padded img shape:\n{padded_img.shape} to be same as "
        f"scaled patch size:\n {tuple(scaled_patch_size)}"
    )

    assert padded_img.shape == padded_seg.shape == padded_bbox_chan.shape
    return dict(
        image=padded_img,
        segmentation=padded_seg,
        bbox_chan=padded_bbox_chan,
        scaled_orig_bbox=scaled_orig_bbox,
        scaled_slicer=scaled_slicer
    )


class CropAndAddBBox:
    def __call__(self, image, segmentation):
        assert segmentation.dtype == bool
        patch_sizes = np.array([192, 192, 192])
        if np.any(segmentation):
            bbox_dict = mask3D_to_bbox(segmentation)

            # make into simpler 3 x 2 (dims x start/stop) numpy array for later calculations
            bbox = np.array(
                [
                    [
                        bbox_dict["z_min"],
                        bbox_dict["z_max"],
                    ],
                    [
                        bbox_dict["z_mid_y_min"],
                        bbox_dict["z_mid_y_max"],
                    ],
                    [
                        bbox_dict["z_mid_x_min"],
                        bbox_dict["z_mid_x_max"],
                    ],
                ]
            )
            image_seg_bbox = crop_and_add_bbox(image, bbox, segmentation, patch_sizes)
            padded_img = image_seg_bbox["image"]
            padded_seg = image_seg_bbox["segmentation"]
            padded_bbox_chan = image_seg_bbox["bbox_chan"]
        else:
            # let's just do center crop of empty image
            n_pad = np.clip(patch_sizes - np.array(image.shape), a_min=0, a_max=None)
            n_pad_before = n_pad // 2
            n_pad_after = n_pad - n_pad_before
            padded_img = np.pad(
                image,
                tuple((n_a, n_b) for n_a, n_b in zip(n_pad_before, n_pad_after)),
            )
            i_starts = (
                np.array(padded_img.shape) // 2 - patch_sizes // 2 - (patch_sizes % 2)
            )
            i_starts = np.clip(i_starts, a_min=0, a_max=None).astype(np.int32)
            i_stops = i_starts + patch_sizes
            padded_img = padded_img[
                i_starts[0] : i_stops[0],
                i_starts[1] : i_stops[1],
                i_starts[2] : i_stops[2],
            ]
            padded_seg = np.zeros_like(padded_img).astype(bool)
            padded_bbox_chan = np.zeros_like(padded_img).astype(bool)
            assert padded_img.shape == tuple(patch_sizes)

        assert padded_img.shape == padded_seg.shape == padded_bbox_chan.shape
        return dict(
            image=padded_img,
            segmentation=padded_seg,
            bbox_chan=padded_bbox_chan,
        )


class MergeImageAndInteractionAndSeg:
    def __call__(self, **data_dict):
        image = data_dict["image"]
        segmentation = data_dict["segmentation"]
        bbox_chan = data_dict["bbox_chan"]
        # img, 1 prompt chan, bbox chan, 5 prompt chans, seg
        full_image = torch.cat(
            (image, torch.zeros_like(image), bbox_chan)
            + tuple(torch.zeros_like(image) for _ in range(5))
            + (segmentation,)
        )
        data_dict["image"] = full_image
        return data_dict
