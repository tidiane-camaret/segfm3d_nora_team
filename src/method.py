import os
import time
import sys
import numpy as np
import torch
import contextlib
import gc
from nnunetv2.utilities.helpers import empty_cache
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from src.training.transforms import (
    crop_to_nonzero_image,
    bbox_to_slicer,
    get_nonzero_bbox,
)
from src.training.transforms import NormalizeSingleImageTransformNumpy
from src.training.transforms import crop_and_add_bbox
from src.pred_util import paste_tensor_leading_dim
from copy import deepcopy
import traceback


def compute_background_log_prob(logits):
    """
    Args:
        logits: Tensor of shape (K, D, H, W) -- these are logits for K foreground classes
                (i.e., values before sigmoid)
    Returns:
        bg_prob: Tensor of shape (D, H, W) with probability of background class
    """
    # Compute sigmoid probabilities
    probs = torch.sigmoid(logits)  # (K, D, H, W)

    # Numerically stable: sum log(1 - p) instead of product
    log_bg_prob = torch.sum(torch.log1p(-probs), dim=0)  # log(1 - p) is log1p(-p)

    return log_bg_prob


class SimplePredictor:
    """
    Predictor class using nnInteractive "as is".
    """

    def __init__(self, checkpoint_path, device, include_previous_clicks, n_pred_iters):
        torch.set_grad_enabled(False)
        self.init_network(checkpoint_path, device)
        self.include_previous_clicks = include_previous_clicks
        self.device = torch.device(device)
        self.click_radius = 4
        assert n_pred_iters > 0
        self.n_pred_iters = n_pred_iters
        print(f"Using {self.n_pred_iters} predictions...")

    def init_network(self, checkpoint_path, device):
        args_for_network = {
            "arch_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
            "arch_kwargs": {
                "n_stages": 6,
                "features_per_stage": [32, 64, 128, 256, 320, 320],
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "kernel_sizes": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "strides": [
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
                "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
                "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
                "conv_bias": True,
                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            "arch_kwargs_req_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            "input_channels": 8,
            "output_channels": 2,
            "allow_init": True,
            "deep_supervision": True,
        }
        network = get_network_from_plans(**args_for_network)  # .cuda()
        print(f"Loading {checkpoint_path}")
        chkpt_dict = torch.load(checkpoint_path, weights_only=False)
        network.load_state_dict(chkpt_dict["network_weights"])
        self.network = network.to(device)

    def predict(
        self,
        image,  # Full 3D image (NumPy ZYX)
        bboxs=None,  # BBox list (used in iter 0) [{'z_min':..,}, ...] * num_classes
        clicks=None,  # Click tuple (used in iter 1+) containing :
        # clicks_cls : coordinates of every fg and bg click so far, {'fg':[[x,y,z],..], 'bg':[[x,y,z],..]} * num_classes
        # clicks_order : order of clicks,  ['fg', 'fg', 'bg' ...] * num_classes
        prev_pred=None,  # Full prediction from previous step (NumPy ZYX)
        num_classes_max=None,  # Optional: limit number of classes processed
    ):
        start_time = time.time()
        forward_pass_time = 0
        crop_scaled_bbox_time = 0

        device = self.device

        ordered_clicks_per_class = []
        forward_pass_count = 0
        for class_clicks, class_order in zip(clicks[0], clicks[1]):
            this_clicks = deepcopy(class_clicks)
            click_dicts = []

            for fg_or_bg in class_order:
                coord = np.array(this_clicks[fg_or_bg].pop(0))
                click_dicts.append(dict(fg_or_bg=fg_or_bg, coord=coord))
            ordered_clicks_per_class.append(click_dicts)
            # assert len(this_clicks["bg"]) == 0
            # assert len(this_clicks["fg"]) == 0

        patch_sizes = np.array([192, 192, 192])

        all_preds = []


        nonzero_bbox = get_nonzero_bbox(image)
        nonzero_slicer = bbox_to_slicer(nonzero_bbox)
        nonzero_image = image[nonzero_slicer]
        normed_image = NormalizeSingleImageTransformNumpy()(image=nonzero_image)[
            "image"
        ].astype(np.float32)
        if bboxs is None:
            # make box in middle third of the image
            bboxs = [
                dict(
                    z_min=(2 * nonzero_bbox[0, 0] + nonzero_bbox[0, 1]) // 3,
                    z_max=(nonzero_bbox[0, 0] + 2 * nonzero_bbox[0, 1]) // 3,
                    z_mid=(nonzero_bbox[0, 0] + nonzero_bbox[0, 1]) // 2,
                    z_mid_y_min=(2 * nonzero_bbox[1, 0] + nonzero_bbox[1, 1]) // 3,
                    z_mid_y_max=(nonzero_bbox[1, 0] + 2 * nonzero_bbox[1, 1]) // 3,
                    z_mid_x_min=(2 * nonzero_bbox[2, 0] + nonzero_bbox[2, 1]) // 3,
                    z_mid_x_max=(nonzero_bbox[2, 0] + 2 * nonzero_bbox[2, 1]) // 3,
                )
            ]

        n_classes = len(bboxs)
        if num_classes_max is not None:
            n_classes = min(n_classes, num_classes_max)
        for i_class in range(n_classes):
            try:
                prev_seg = prev_pred == (i_class + 1)
                nonzero_prev_seg = prev_seg[nonzero_slicer]
                bbox_dict = bboxs[i_class]
                class_bbox = np.array(
                    [
                        [bbox_dict["z_min"], bbox_dict["z_max"]],
                        [bbox_dict["z_mid_y_min"], bbox_dict["z_mid_y_max"]],
                        [bbox_dict["z_mid_x_min"], bbox_dict["z_mid_x_max"]],
                    ]
                )

                class_bbox_in_nonzero = class_bbox - nonzero_bbox[:, 0:1]

                all_clicks_this_class = ordered_clicks_per_class[i_class]

                (
                    net_pred,
                    scaled_bbox_in_full_image,
                    this_crop_scaled_bbox_time,
                    this_forward_pass_time,
                ) = self.crop_and_pred(
                    normed_image=normed_image,
                    nonzero_bbox=nonzero_bbox,
                    class_bbox_in_nonzero=class_bbox_in_nonzero,
                    nonzero_prev_seg=nonzero_prev_seg,
                    patch_sizes=patch_sizes,
                    all_clicks_this_class=all_clicks_this_class,
                    extra_bbox_for_crop_in_nonzero=None,
                )
                

                this_pred_in_image = torch.full(
                    (net_pred.shape[0],) + image.shape, torch.nan, device=net_pred.device
                )

                forward_pass_count += 1
                # _ = net_pred.cpu().numpy() # just for time checking
                forward_pass_time += this_forward_pass_time
                crop_scaled_bbox_time += this_crop_scaled_bbox_time

                pasted_pred = paste_tensor_leading_dim(
                    this_pred_in_image, net_pred, bbox=scaled_bbox_in_full_image
                )
                del net_pred
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    empty_cache(torch.device("cuda", 0))
                gc.collect()
                # you could make a loop here if you want
                for _ in range(1, self.n_pred_iters):
                    nonzero_prev_seg = (np.diff(pasted_pred.cpu().numpy(), axis=0) > 0).squeeze(0)[
                        nonzero_slicer
                    ]

                    (
                        net_pred,
                        scaled_bbox_in_full_image,
                        this_crop_scaled_bbox_time,
                        this_forward_pass_time,
                    ) = self.crop_and_pred(
                        normed_image=normed_image,
                        nonzero_bbox=nonzero_bbox,
                        class_bbox_in_nonzero=class_bbox_in_nonzero,
                        nonzero_prev_seg=nonzero_prev_seg,
                        patch_sizes=patch_sizes,
                        all_clicks_this_class=all_clicks_this_class,
                        extra_bbox_for_crop_in_nonzero=None,
                    )
                    forward_pass_count += 1
                    # _ = net_pred.cpu().numpy() # just for time checking
                    forward_pass_time += this_forward_pass_time
                    crop_scaled_bbox_time += this_crop_scaled_bbox_time

                
                    pasted_pred = paste_tensor_leading_dim(
                        pasted_pred, net_pred, bbox=scaled_bbox_in_full_image
                    )
                    del net_pred
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        empty_cache(torch.device("cuda", 0))
                    gc.collect()


                all_preds.append(pasted_pred)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    empty_cache(torch.device("cuda", 0))
                gc.collect()

            except Exception as e:
                print(f"Error on class {i_class + 1}:", e)
                traceback.print_exc()
                fake_pred = torch.zeros((2,) + image.shape, device=device)
                # predict this class as not existing
                fake_pred[:, 1] = -torch.inf
                all_preds.append(fake_pred)
                break

        logits_per_class = torch.nan_to_num(
            torch.diff(torch.stack(all_preds), axis=1)[:, 0], -torch.inf
        )
        bg_log_prob_1 = compute_background_log_prob(logits_per_class)
        logits_with_background = torch.cat((bg_log_prob_1[None], logits_per_class))
        full_seg = logits_with_background.argmax(dim=0).cpu().numpy()

        total_time = time.time() - start_time
        print(
            f"Finished predicting in {total_time:.2}, {forward_pass_time:.2} in forward and rescale ({(forward_pass_time / total_time):.1%}),",
            f"{crop_scaled_bbox_time:.2} in cropping with bbox ({(crop_scaled_bbox_time / total_time):.1%})",
        )

        prediction_metrics = {
            "infer_time": total_time,
            "forward_pass_count": forward_pass_count,
        }
        return full_seg, prediction_metrics


    def crop_and_pred(
        self,
        normed_image,
        nonzero_bbox,
        class_bbox_in_nonzero,
        nonzero_prev_seg,
        patch_sizes,
        all_clicks_this_class,
        extra_bbox_for_crop_in_nonzero,
    ):
        device = self.device

        forward_pass_time = 0
        # abuse a bit the fact here we have two images that go into this function
        # "image" and "segmentation"
        start_time_crop_scaled_bbox = time.time()

        if extra_bbox_for_crop_in_nonzero is None:
            bbox_for_crop = class_bbox_in_nonzero
        else:
            bbox_for_crop = extra_bbox_for_crop_in_nonzero

        cropped_im_and_scaled_bbox = crop_and_add_bbox(
            image=normed_image,
            bbox=bbox_for_crop,
            segmentation=nonzero_prev_seg,
            patch_sizes=patch_sizes,
        )
        if extra_bbox_for_crop_in_nonzero is None:
            class_cropped_bbox_chan = cropped_im_and_scaled_bbox["bbox_chan"]
        else:  # now bbox may be at arbitrary position
            # so we just will push the chan through
            nonzero_bbox_chan = np.zeros(normed_image.shape, dtype=bool)
            # fill bbox
            nonzero_bbox_chan[bbox_to_slicer(class_bbox_in_nonzero)] = 1
            class_cropped_bbox_chan = crop_and_add_bbox(
                image=normed_image,
                bbox=bbox_for_crop,
                segmentation=nonzero_bbox_chan,
                patch_sizes=patch_sizes,
            )["segmentation"]

        crop_scaled_bbox_time = time.time() - start_time_crop_scaled_bbox

        class_cropped_im = cropped_im_and_scaled_bbox["image"]
        class_cropped_prev_seg = cropped_im_and_scaled_bbox["segmentation"]
        scaled_bbox_in_nonzero = cropped_im_and_scaled_bbox["scaled_orig_bbox"]
        del cropped_im_and_scaled_bbox

        scaled_bbox_in_full_image = scaled_bbox_in_nonzero + nonzero_bbox[:, 0:1]
        # scale should be same throughout
        scaled_bbox_size = np.diff(scaled_bbox_in_full_image, axis=1).squeeze(1)
        scaled_bbox_scale = scaled_bbox_size / patch_sizes
        assert np.all(scaled_bbox_scale == scaled_bbox_scale[0])

        resized_image = torch.nn.functional.interpolate(
            torch.tensor(class_cropped_im[None, None]).to(device),
            size=tuple(patch_sizes),
            mode="trilinear",
        ).squeeze(0)
        resized_bbox_chan = torch.nn.functional.interpolate(
            torch.tensor(class_cropped_bbox_chan[None, None]).float().to(device),
            size=tuple(patch_sizes),
            mode="nearest",
        ).squeeze(0)
        resized_prev_seg = torch.nn.functional.interpolate(
            torch.tensor(class_cropped_prev_seg[None, None]).float().to(device),
            size=tuple(patch_sizes),
            mode="nearest",
        ).squeeze(0)

        # compute clicks
        pos_click_chan = torch.zeros_like(resized_bbox_chan, dtype=bool, device=device)
        neg_click_chan = torch.zeros_like(resized_bbox_chan, dtype=bool, device=device)

        if self.include_previous_clicks:
            wanted_clicks = all_clicks_this_class
        else:
            wanted_clicks = all_clicks_this_class[-1:]

        for one_click in wanted_clicks:
            click_coord = one_click["coord"]
            click_fg_or_bg = one_click["fg_or_bg"]

            click_coord_in_bbox = (
                np.array(click_coord) - scaled_bbox_in_full_image[:, 0]
            )

            click_coord_scaled = click_coord_in_bbox / scaled_bbox_scale

            z_indices, y_indices, x_indices = torch.meshgrid(
                torch.arange(resized_bbox_chan.shape[1], device=device),
                torch.arange(resized_bbox_chan.shape[2], device=device),
                torch.arange(resized_bbox_chan.shape[3], device=device),
                indexing="ij",
            )
            diffs = (
                torch.stack((z_indices, y_indices, x_indices))
                - torch.tensor(click_coord_scaled, device=device)[:, None, None, None]
            )
            clicks_mask = (
                torch.sqrt(torch.sum(diffs**2, dim=0, keepdim=True)) < self.click_radius
            )
            del z_indices, y_indices, x_indices, diffs

            if click_fg_or_bg == "fg":
                pos_click_chan = pos_click_chan | clicks_mask
            else:
                assert click_fg_or_bg == "bg"
                neg_click_chan = neg_click_chan | clicks_mask
        zero_chan = torch.zeros_like(resized_bbox_chan).to(device)
        input_for_net = torch.cat(
            (
                resized_image.to(device),
                resized_prev_seg.to(device),
                resized_bbox_chan.to(device),
                zero_chan,
                pos_click_chan,
                neg_click_chan,
                zero_chan,
                zero_chan,
            )
        ).to(device)
        del resized_image, resized_prev_seg, resized_bbox_chan, zero_chan, pos_click_chan, neg_click_chan
        start_forward = time.time()
        net_pred = self.network(input_for_net[None])[0][0]
        forward_pass_time = time.time() - start_forward
        if not np.all(np.array(net_pred.shape[1:]) == scaled_bbox_size):
            print("Rescaling pred")
            net_pred = torch.nn.functional.interpolate(
                net_pred[None], list(scaled_bbox_size), mode="trilinear"
            )[0]

        return (
            net_pred,
            scaled_bbox_in_full_image,
            crop_scaled_bbox_time,
            forward_pass_time,
        )
