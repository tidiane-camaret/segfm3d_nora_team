from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
import os.path
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from acvl_utils.cropping_and_padding.bounding_boxes import (
    bounding_box_to_slice,
    crop_and_pad_nd,
)
from nnInteractive.utils.crop import (
    crop_and_pad_into_buffer,
    paste_tensor,
    pad_cropped,
    crop_to_valid,
)
from torch.nn.functional import interpolate
from typing import Union, List, Tuple, Optional
import torch


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


class NPZPytorchDataset(Dataset):
    def __init__(self, npz_dataset, transform, fixed_class=None):
        self.npz_dataset = npz_dataset
        self.transform = transform
        self.fixed_class = fixed_class

    def __getitem__(self, idx):
        patch_size = (192, 192, 192)
        img_cropped, seg_cropped, seg_prev, properties = self.npz_dataset[
            self.npz_dataset.identifiers[idx]
        ]
        class_locations = properties["class_locations"]
        if len(class_locations) > 0:
            classes_ints = [
                key for key in class_locations.keys() if len(class_locations[key]) > 0
            ]
            if self.fixed_class is None:
                wanted_class_idx = np.random.choice(list(classes_ints))
            else:
                wanted_class_idx = self.fixed_class
            this_coords = class_locations[wanted_class_idx]
        else:
            # background class only there
            wanted_class_idx = 0
            this_coords = np.argwhere(seg_cropped == 0)

        i_starts, i_ends = np.min(this_coords, axis=0), np.max(this_coords, axis=0)
        bbox_tuple = list(zip(i_starts[1:], i_ends[1:]))
        preproced_image, preprocessed_props = preproc_image(img_cropped)  #
        bbox_preproc = preprocessed_props["bbox_used_for_cropping"]
        preproced_seg = torch.tensor(
            seg_cropped[
                :,
                bbox_preproc[0][0] : bbox_preproc[0][1],
                bbox_preproc[1][0] : bbox_preproc[1][1],
                bbox_preproc[2][0] : bbox_preproc[2][1],
            ]
            == wanted_class_idx
        )

        assert preproced_seg.shape == preproced_image.shape
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox_tuple]
        bbox_size = [i[1] - i[0] for i in bbox_tuple]

        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, patch_size)]
        zoom_out_factors_from_bbox = max(
            1, max([i / j for i, j in zip(requested_size, patch_size)])
        )

        prediction_center = bbox_center
        zoom_out_factor = min(zoom_out_factors_from_bbox, 4)
        # initial prediction at initial_zoom_out_factor
        scaled_patch_size = [round(p_size * zoom_out_factor) for p_size in patch_size]
        scaled_bbox = [
            [c - p // 2, c + p // 2 + p % 2]
            for c, p in zip(prediction_center, scaled_patch_size)
        ]
        crop_img, _ = crop_to_valid(preproced_image, scaled_bbox)
        crop_seg, _ = crop_to_valid(preproced_seg, scaled_bbox)

        return dict(
            **self.transform(image=crop_img.float(), segmentation=crop_seg),
            identifier=properties["identifier"],
            target_class=wanted_class_idx,
        )

    def __len__(self):
        return len(self.npz_dataset.identifiers)


class NPZDataset(nnUNetBaseDataset):
    def __init__(self, orig_folder, save_folder, files):
        self.orig_folder = orig_folder
        self.save_folder = save_folder
        self.identifiers = files

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        assert identifier.endswith(".npz")
        saved_file_path = os.path.join(self.save_folder, identifier)
        # first preproc and save
        if not os.path.exists(saved_file_path):
            image_and_gts = np.load(os.path.join(self.orig_folder, identifier))
            seg = image_and_gts["gts"]
            img = image_and_gts["imgs"]
            shape_before_cropping = seg.shape
            assert img.shape == seg.shape

            valid_mask = seg > 0
            coords_valid = np.argwhere(valid_mask)
            bbox = np.array(
                list(
                    zip(
                        np.min(coords_valid, axis=0), np.max(coords_valid, axis=0) + 1
                    )  # + 1 since python indices are exclusive
                )
            )
            seg_cropped = seg[
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
            ]
            img_cropped = img[
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
            ]
            shape_after_cropping_and_before_resampling = img_cropped.shape
            n_classes = np.max(seg_cropped) + 1
            class_locations = dict()
            for i_class in range(1, n_classes):
                class_locations[i_class] = np.argwhere(seg_cropped[None] == i_class)

            affine = np.zeros((4, 4))
            np.fill_diagonal(affine, list(image_and_gts["spacing"]) + [1.0])
            nibabel_stuff = dict(original_affine=affine, reoriented_affine=affine)
            spacing = image_and_gts["spacing"]
            properties = dict(
                nibabel_stuff=nibabel_stuff,
                spacing=spacing,
                shape_before_cropping=shape_before_cropping,
                bbox_used_for_cropping=[
                    [0, shape_before_cropping[0]],
                    [0, shape_before_cropping[1]],
                    [0, shape_before_cropping[2]],
                ],
                shape_after_cropping_and_before_resampling=shape_after_cropping_and_before_resampling,
                # class_locations=class_locations,
            )
            # ensure folder exists
            os.makedirs(Path(saved_file_path).parent, exist_ok=True)
            np.savez_compressed(
                saved_file_path,
                img_cropped=img_cropped,
                seg_cropped=seg_cropped,
                properties=properties,
                **{str(i): a for i, a in class_locations.items()},
            )
        loaded_npz = np.load(saved_file_path, allow_pickle=True)
        img_cropped = loaded_npz["img_cropped"]
        seg_cropped = loaded_npz["seg_cropped"]
        properties = loaded_npz["properties"].item()
        i_classes = sorted(
            int(key)
            for key in loaded_npz.keys()
            if key not in ["img_cropped", "seg_cropped", "properties"]
        )
        class_locations = {i: loaded_npz[str(i)] for i in i_classes}
        properties["class_locations"] = class_locations
        properties["identifier"] = identifier
        seg_prev = None
        return img_cropped[None], seg_cropped[None], seg_prev, properties

    def get_identifiers():
        raise NotImplementedError("not implemented")

    def save_case():
        raise NotImplementedError("not implemented")
