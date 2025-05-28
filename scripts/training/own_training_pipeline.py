import os
import time
import sys
import numpy as np
import torch
import contextlib
import gc
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import os.path


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
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
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

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
network = get_network_from_plans(**args_for_network).cuda()
orig_checkpoint_dir = '/nfs/norasys/notebooks/camaret/model_checkpoints/nnint/nnInteractive_v1.0/'
newer_checkpoint_dir = "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_segfm-robin/data/model-checkpoints/try_from_nb_all_data_competition_bbox/"
if os.path.exists(newer_checkpoint_dir):
    chkpt_dict = torch.load(os.path.join(newer_checkpoint_dir, 'fold_0/checkpoint_final.pth'), weights_only=False)
else:
    chkpt_dict = torch.load(os.path.join(orig_checkpoint_dir, 'fold_0/checkpoint_final.pth'), weights_only=False)

network.load_state_dict(chkpt_dict['network_weights'])

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import (
    ApplyRandomBinaryOperatorTransform,
)
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import (
    MoveSegAsOneHotToDataTransform,
)
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert3DTo2DTransform,
    Convert2DTo3DTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import (
    ConvertSegmentationToRegionsTransform,
)
from src.training.transforms import AddSegToImageTransform
from src.training.transforms import NormalizeSingleImageTransform
from src.training.transforms import AddBBoxAndEmptyChannelsSingleClassTransform
from src.training.transforms import MONAIRandSpatialTransform
from src.training.transforms import MONAIFixedSpatialTransform


only_2d_bbox = False

train_transform = ComposeTransforms(
    [
        NormalizeSingleImageTransform(),
        MONAIRandSpatialTransform(),
        RandomTransform(
            apply_probability=0.1,
            transform=GaussianNoiseTransform(
                noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True
            ),
        ),
        RandomTransform(
            apply_probability=0.2,
            transform=GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                benchmark=True,
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
            ),
        ),
        RandomTransform(
            apply_probability=0.15,
            transform=MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast(contrast_range=(0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1,
            ),
        ),
        RandomTransform(
            apply_probability=0.15,
            transform=ContrastTransform(
                contrast_range=BGContrast(contrast_range=(0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            ),
        ),
        RandomTransform(
            apply_probability=0.25,
            transform=SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
        ),
        RandomTransform(
            apply_probability=0.1,
            transform=GammaTransform(
                gamma=BGContrast(contrast_range=(0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
        ),
        RandomTransform(
            apply_probability=0.3,
            transform=GammaTransform(
                gamma=BGContrast(contrast_range=(0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
        ),
        MirrorTransform(allowed_axes=(0, 1, 2)),
        RemoveLabelTansform(segmentation_channels=None, label_value=-1, set_to=0),
        DownsampleSegForDSTransform(
            ds_scales=[
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.125, 0.125, 0.125],
                [0.0625, 0.0625, 0.0625],
            ]
        ),
        AddBBoxAndEmptyChannelsSingleClassTransform(only_2d_bbox=only_2d_bbox),
        AddSegToImageTransform(),
    ]
)
test_transform = ComposeTransforms(
    [
        NormalizeSingleImageTransform(),
        MONAIFixedSpatialTransform(),
        RemoveLabelTansform(segmentation_channels=None, label_value=-1, set_to=0),
        AddBBoxAndEmptyChannelsSingleClassTransform(only_2d_bbox=only_2d_bbox),
        AddSegToImageTransform(),
    ]
)


from src.training.npzdataset import NPZPytorchDataset, NPZDataset
from src.training.utils import fast_ids_to_deterministic_floats
import numpy as np

from glob import glob
parent_folder = '/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/SegFM3D/data/3D_train_npz_random_10percent_16G/'
parent_save_folder = '/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/SegFM3D/data/own-preproc/'

all_npz_file_paths = glob(os.path.join(parent_folder, '**/*.npz'), recursive=True)
identifiers = np.array([f.replace(parent_folder, "") for f in all_npz_file_paths])

train_mask = fast_ids_to_deterministic_floats(identifiers) < 0.8
train_files = identifiers[train_mask]
test_files = identifiers[~train_mask]
dataset_tr = NPZDataset(parent_folder, parent_save_folder, train_files)
dataset_val = NPZDataset(parent_folder, parent_save_folder, test_files)
train_set = NPZPytorchDataset(
    dataset_tr, transform=train_transform)

test_set = NPZPytorchDataset(
    dataset_val, transform=test_transform)


n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])

import torch
train_loader = torch.utils.data.DataLoader(
    train_set,
    shuffle=True,
    drop_last=True,
    num_workers=n_jobs,
    batch_size=2,
)

test_set = NPZPytorchDataset(
    dataset_val, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_set,
    shuffle=False,
    drop_last=False,
    num_workers=n_jobs,
    batch_size=2,
)


from tqdm.autonotebook import tqdm, trange
import torch
from src.training.model_wrap import ModelPrevSegAndClickWrapper
from src.training.utils import defer_keysignal_with_grad

wrapped_network = ModelPrevSegAndClickWrapper(network, n_max_clicks=1)

optim_network = torch.optim.AdamW(
    wrapped_network.parameters(), lr=3e-4, weight_decay=1e-5
)


from copy import deepcopy
import shutil
out_dir = '/nfs/norasys/notebooks/camaret/model_checkpoints/nnint/finetuned_28_05'
try:
    
    shutil.copytree(orig_checkpoint_dir, out_dir)
except FileExistsError:
    print(f"path {out_dir} exists already, fine")
trained_chkpt_path = os.path.join(out_dir, 'fold_0/checkpoint_final.pth')
assert os.path.exists(trained_chkpt_path)


trained_chkpt = deepcopy(chkpt_dict)


from src.training.utils import compute_binary_dsc
from IPython.display import display
import pandas as pd
n_epochs = 25
moving_dsc = None
for i_epoch in trange(n_epochs):
    all_mean_dscs = []
    for batch in (pbar := tqdm(train_loader)):
        with defer_keysignal_with_grad():
            targets = batch["segmentation"][0][:,0].long().cuda()
            out = wrapped_network(batch["image"].cuda())
            # assuming deep supervision
            pred = out[0]
            assert pred[:,0].shape == targets.shape
            cent = torch.nn.functional.cross_entropy(pred, targets)
            binary_pred = torch.sigmoid(torch.diff(pred, dim=1)[:,0])
            assert binary_pred.shape == targets.shape
            eps = 1e-3
            intersect = torch.sum(binary_pred * targets, dim=(1,2,3))
            sum_pred = torch.sum(binary_pred, dim=(1,2,3))
            sum_targets = torch.sum(targets, dim=(1,2,3))
            mean_dsc = torch.mean((2 * intersect + eps) / (sum_pred + sum_targets + eps))
            loss = cent + 10 * (1 - mean_dsc)
            optim_network.zero_grad()
            loss.backward()
            if torch.isfinite(loss).item():
                optim_network.step()
            else:
                print("loss not finite, not updating!")
            optim_network.zero_grad()
        all_mean_dscs.append(mean_dsc.item())
        if torch.isfinite(loss).item():
            moving_dsc = mean_dsc if moving_dsc is None else (moving_dsc * 0.98 + mean_dsc * 0.02)
        pbar.set_postfix(dict(moving_dsc=moving_dsc.item()))
        print(mean_dsc.item())
        print(cent.item())
        print()
    print("mean dsc", np.mean(all_mean_dscs))
    
    print(f"Epoch {i_epoch}")
    if torch.isfinite(loss).item():
        trained_chkpt['network_weights'] = wrapped_network.orig_network.state_dict()
        torch.save(trained_chkpt, trained_chkpt_path)
        print(f"Checkpoint saved to {trained_chkpt_path}")

    test_results = []
    for batch in (pbar := tqdm(test_loader)):
        targets = batch["segmentation"][:, 0].long().cuda()
        out = wrapped_network(batch["image"].cuda())
        # assuming deep supervision
        pred = out[0]
        assert pred[:, 0].shape == targets.shape
        cent = torch.nn.functional.cross_entropy(pred, targets)
        binary_pred = torch.sigmoid(torch.diff(pred, dim=1)[:, 0])
        assert binary_pred.shape == targets.shape
        eps = 1e-3
        intersect = torch.sum(binary_pred * targets, dim=(1, 2, 3))
        sum_pred = torch.sum(binary_pred, dim=(1, 2, 3))
        sum_targets = torch.sum(targets, dim=(1, 2, 3))
        mean_dsc = torch.mean((2 * intersect + eps) / (sum_pred + sum_targets + eps))
        test_results.append(
            dict(
                hard_dsc=compute_binary_dsc((binary_pred > 0.5), targets),
                soft_dsc=((2 * intersect + eps) / (sum_pred + sum_targets + eps)).cpu().numpy().mean(),
                cent=cent.item()
            )
        )
    display(pd.DataFrame(test_results).mean())


    test_results = []
for batch in (pbar := tqdm(test_loader)):
    targets = batch["segmentation"][:, 0].long().cuda()
    out = wrapped_network(batch["image"].cuda())
    # assuming deep supervision
    pred = out[0]
    assert pred[:, 0].shape == targets.shape
    cent = torch.nn.functional.cross_entropy(pred, targets)
    binary_pred = torch.sigmoid(torch.diff(pred, dim=1)[:, 0])
    assert binary_pred.shape == targets.shape
    eps = 1e-3
    intersect = torch.sum(binary_pred * targets, dim=(1, 2, 3))
    sum_pred = torch.sum(binary_pred, dim=(1, 2, 3))
    sum_targets = torch.sum(targets, dim=(1, 2, 3))
    mean_dsc = torch.mean((2 * intersect + eps) / (sum_pred + sum_targets + eps))
    test_results.append(
        dict(
            hard_dsc=compute_binary_dsc((binary_pred > 0.5), targets),
            soft_dsc=((2 * intersect + eps) / (sum_pred + sum_targets + eps)).cpu().numpy().mean(),
            cent=cent.item()
        )
    )