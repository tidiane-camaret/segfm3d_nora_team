import os

import torch
from src.config import config

n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 8
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
orig_checkpoint_dir = os.path.join(config["NNINT_CKPT_DIR"], "nnInteractive_v1.0")

chkpt_dict = torch.load(
    os.path.join(orig_checkpoint_dir, "fold_0/checkpoint_final.pth"), weights_only=False
)

network.load_state_dict(chkpt_dict["network_weights"])


from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from monai.transforms import Resized
from src.training.npzdataset import NPZNoCacheDataset
from src.training.transforms import (
    AddEmptyChan,
    ChooseClass,
    CropAndAddBBox,
    CropToNonzeroImage,
    MergeImageAndInteractionAndSeg,
    NormalizeSingleImageTransformNumpy,
    WrapMONAI,
)

only_2d_bbox = False

test_transforms = [
    CropToNonzeroImage(),
    NormalizeSingleImageTransformNumpy(),
    ChooseClass(),
    CropAndAddBBox(),
    AddEmptyChan(),
    WrapMONAI(
        Resized(
            keys=["image", "segmentation", "bbox_chan"],
            spatial_size=(192, 192, 192),
            mode=["trilinear", "nearest", "nearest"],
        )
    ),
    MergeImageAndInteractionAndSeg(),
]
test_transform = ComposeTransforms(test_transforms)
train_transforms = test_transforms + [
    DownsampleSegForDSTransform(
        ds_scales=[
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.125, 0.125, 0.125],
            [0.0625, 0.0625, 0.0625],
        ]
    )
]
train_transform = ComposeTransforms(train_transforms)

from glob import glob

import numpy as np
from src.training.npzdataset import NPZDataset, NPZPytorchDataset
from src.training.utils import fast_ids_to_deterministic_floats

parent_folder = os.path.join(config["DATA_DIR"], "3D_train_npz_random_10percent_16G")

to_exclude = [
    "Ultrasound/US_Low-limb-Leg/US_Low-limb-Leg07.npz",
    "Ultrasound/US_Low-limb-Leg/US_Low-limb-Leg24.npz",
    "Ultrasound/US_Low-limb-Leg/US_Low-limb-Leg37.npz",
    "CT/CT_Abdomen1K/CT_Abdomen1K_Case_00931.npz",
    "Microscopy/Microscopy_cremi/Microscopy_cremi_002_sc.npz",
    "Microscopy/Microscopy_cremi/Microscopy_cremi_000_sc.npz",
    "Microscopy/Microscopy_cremi/Microscopy_cremi_001_sc.npz",
]
all_npz_file_paths = glob(os.path.join(parent_folder, "**/*.npz"), recursive=True)

cleaned_file_paths = np.array(
    sorted([f for f in all_npz_file_paths if not any(e in f for e in to_exclude)])
)

train_mask = fast_ids_to_deterministic_floats(cleaned_file_paths) < 0.8
train_files = cleaned_file_paths[train_mask]
test_files = cleaned_file_paths[~train_mask]
train_set = NPZNoCacheDataset(train_files, transform=train_transform)
test_set = NPZNoCacheDataset(test_files, transform=test_transform)


train_loader = torch.utils.data.DataLoader(
    train_set,
    shuffle=True,
    drop_last=True,
    num_workers=n_jobs,
    batch_size=2,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    shuffle=False,
    drop_last=False,
    num_workers=n_jobs,
    batch_size=2,
)


import torch
from src.training.model_wrap import ModelPrevSegAndClickWrapper
from src.training.utils import defer_keysignal_with_grad
from tqdm.autonotebook import tqdm, trange

wrapped_network = ModelPrevSegAndClickWrapper(network, n_max_clicks=1)

optim_network = torch.optim.AdamW(
    wrapped_network.parameters(), lr=1e-4, weight_decay=1e-5
)


import shutil
from copy import deepcopy

out_dir = os.path.join(config["RESULTS_DIR"], "trained_model_segfm3d_nora_team")
try:
    shutil.copytree(orig_checkpoint_dir, out_dir)
except FileExistsError:
    print(f"path {out_dir} exists already, fine")
trained_chkpt_path = os.path.join(out_dir, "fold_0/checkpoint_final.pth")
assert os.path.exists(trained_chkpt_path)


trained_chkpt = deepcopy(chkpt_dict)


accumulate_batch_size = 16
n_batches_accumulate = accumulate_batch_size // 2


from datetime import datetime

import pandas as pd
from src.training.utils import compute_binary_dsc

n_epochs = 25
moving_dsc = None
all_mean_dscs = []
i_batch = 0
optim_network.zero_grad()
for i_epoch in trange(n_epochs):
    epoch_mean_dscs = []
    for batch in (pbar := tqdm(train_loader)):
        with defer_keysignal_with_grad():
            targets = batch["segmentation"][0][:, 0].long().cuda()
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
            mean_dsc = torch.mean(
                (2 * intersect + eps) / (sum_pred + sum_targets + eps)
            )
            loss = cent + 10 * (1 - mean_dsc)
            loss = loss / n_batches_accumulate  ## account for accumulation
            loss.backward()
            i_batch += 1
            if (i_batch % n_batches_accumulate) == 0:
                all_grad_finite = all(
                    [
                        (p.grad.isfinite().all().item())
                        for group in optim_network.param_groups
                        for p in group["params"]
                        if (p.grad is not None)
                    ]
                )
                if all_grad_finite and torch.isfinite(loss).item():
                    optim_network.step()
                else:
                    print("loss/grads not finite, not updating!")
                optim_network.zero_grad()
            else:
                pass  ## accumulate grad...
        epoch_mean_dscs.append(mean_dsc.item())
        if torch.isfinite(loss).item():
            moving_dsc = (
                mean_dsc
                if moving_dsc is None
                else (moving_dsc * 0.98 + mean_dsc * 0.02)
            )
        pbar.set_postfix(dict(moving_dsc=moving_dsc.item()))
        print(mean_dsc.item())
        print(cent.item())
        print()
    print("mean dsc", np.nanmean(epoch_mean_dscs))
    all_mean_dscs.extend(epoch_mean_dscs)
    print(f"Epoch {i_epoch}")
    if torch.isfinite(loss).item():
        filename_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        trained_chkpt["network_weights"] = wrapped_network.orig_network.state_dict()
        if torch.isfinite(loss).item():
            torch.save(trained_chkpt, trained_chkpt_path)
            print(f"Checkpoint saved to {trained_chkpt_path}")
        current_ckpt_filename = trained_chkpt_path.replace(
            "/checkpoint_final.pth", f"/checkpoint_{i_epoch}_{filename_time_str}.pth"
        )
        torch.save(trained_chkpt, current_ckpt_filename)
        print(f"Checkpoint also saved to {current_ckpt_filename}")

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
            soft_dsc=((2 * intersect + eps) / (sum_pred + sum_targets + eps))
            .cpu()
            .numpy()
            .mean(),
            cent=cent.item(),
        )
    )
display(pd.DataFrame(test_results).mean())
test_result_filepath = os.path.join(
    os.path.split(trained_chkpt_path)[0],
    f"test_results_{i_epoch}_{filename_time_str}.csv",
)
# save test results
pd.DataFrame(test_results).to_csv(test_result_filepath)
print(f"Saved test results to {test_result_filepath}")
