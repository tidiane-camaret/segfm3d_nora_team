import os
import wandb
import pandas as pd
import numpy as np
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from src.config import config
from src.training.utils import args_for_network, test_transform, train_transform
from src.eval import evaluate

n_epochs = 1
lr = 3e-4
weight_decay = 1e-5
n_max_clicks = 1
num_eval_cases = 10

wandb_project_name = "segfm3d_nora_team"  # Align with eval.py's default
wandb.init(
    project=wandb_project_name,
    #name=wandb_run_name,
    config={
        "n_epochs": n_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_max_clicks": n_max_clicks,
    },
)

network = get_network_from_plans(**args_for_network).cuda()
orig_checkpoint_dir = os.path.join(config["NNINT_CKPT_DIR"], "nnInteractive_v1.0/")


chkpt_dict = torch.load(
    os.path.join(orig_checkpoint_dir, "fold_0/checkpoint_final.pth"), weights_only=False
)
network.load_state_dict(chkpt_dict["network_weights"])

from glob import glob

import numpy as np
from src.training.npzdataset import NPZDataset, NPZPytorchDataset
from src.training.utils import fast_ids_to_deterministic_floats

parent_folder = "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/SegFM3D/data/3D_train_npz_random_10percent_16G/"
parent_save_folder = (
    "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/SegFM3D/data/own-preproc/"
)

all_npz_file_paths = glob(os.path.join(parent_folder, "**/*.npz"), recursive=True)
to_exclude = [
    "Ultrasound/US_Low-limb-Leg/US_Low-limb-Leg07.npz",
    "Ultrasound/US_Low-limb-Leg/US_Low-limb-Leg24.npz",
    "Ultrasound/US_Low-limb-Leg/US_Low-limb-Leg37.npz",
    "CT/CT_Abdomen1K/CT_Abdomen1K_Case_00931.npz",
    "Microscopy/Microscopy_cremi/Microscopy_cremi_002_sc.npz",
    "Microscopy/Microscopy_cremi/Microscopy_cremi_000_sc.npz",
    "Microscopy/Microscopy_cremi/Microscopy_cremi_001_sc.npz",
]


identifiers = np.array([f.replace(parent_folder, "") for f in all_npz_file_paths])
identifiers = np.array([f for f in identifiers if f not in to_exclude])

train_mask = fast_ids_to_deterministic_floats(identifiers) < 0.8
train_files = identifiers[train_mask]
test_files = identifiers[~train_mask]
dataset_tr = NPZDataset(parent_folder, parent_save_folder, train_files)
dataset_val = NPZDataset(parent_folder, parent_save_folder, test_files)
train_set = NPZPytorchDataset(dataset_tr, transform=train_transform)

test_set = NPZPytorchDataset(dataset_val, transform=test_transform)

n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])

train_loader = torch.utils.data.DataLoader(
    train_set,
    shuffle=True,
    drop_last=True,
    num_workers=n_jobs,
    batch_size=2,
)

test_set = NPZPytorchDataset(dataset_val, transform=test_transform)
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

wrapped_network = ModelPrevSegAndClickWrapper(network, n_max_clicks=n_max_clicks)

optim_network = torch.optim.AdamW(
    wrapped_network.parameters(), lr=lr, weight_decay=weight_decay
)

from copy import deepcopy
import shutil
out_dir = os.path.join(
    config["NNINT_CKPT_DIR"],
    "nnInteractive_v1.0/finetuned_28_05/",
)
try:
    shutil.copytree(orig_checkpoint_dir, out_dir)
except FileExistsError:
    print(f"path {out_dir} exists already, fine")
trained_chkpt_path = os.path.join(out_dir, 'fold_0/checkpoint_final.pth')
assert os.path.exists(trained_chkpt_path)

# copy the checkpoint dict for the previous segmentation logic
trained_chkpt = deepcopy(chkpt_dict)


from src.training.utils import compute_binary_dsc
moving_dsc = None
all_mean_dscs = []
for i_epoch in trange(n_epochs):
    epoch_mean_dscs = []
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
        epoch_mean_dscs.append(mean_dsc.item())
        if torch.isfinite(loss).item():
            moving_dsc = mean_dsc if moving_dsc is None else (moving_dsc * 0.98 + mean_dsc * 0.02)
        pbar.set_postfix(dict(moving_dsc=moving_dsc.item()))
        print(mean_dsc.item())
        print(cent.item())
        print()
    print("mean dsc", np.nanmean(epoch_mean_dscs))
    all_mean_dscs.extend(epoch_mean_dscs)
    
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

df_test_results = pd.DataFrame(test_results)
mean_test_results = df_test_results.mean()
final_test_hard_dsc = mean_test_results.get('hard_dsc', float('nan'))
final_test_soft_dsc = mean_test_results.get('soft_dsc', float('nan'))
final_test_cent = mean_test_results.get('cent', float('nan'))

wandb.log({
    "training_epochs_completed": n_epochs, # n_epochs is the total planned and executed
    "final_train_test_hard_dsc": final_test_hard_dsc,
    "final_train_test_soft_dsc": final_test_soft_dsc,
    "final_train_test_cent": final_test_cent
})

print(f"\n--- Starting evaluation using eval.py script ---")

# Define paths for evaluation - adjust these if your config structure is different
# or if these paths are not what you intend for evaluation.
# Assuming 'config' is loaded and contains 'DATA_DIR' and 'RESULTS_DIR'
eval_img_dir = os.path.join(config["DATA_DIR"], "3D_val_npz") # Example path
eval_gt_dir = os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg") # Example path
eval_output_dir_base = os.path.join(config["RESULTS_DIR"], "eval_after_own_training")

output_dir = os.path.join(config["RESULTS_DIR"])

# Number of cases to evaluate, set to 0 to run on all cases found by eval.py
# or adjust as needed. eval.py defaults to 10.
evaluate(
    method="nnint_custom",  # This method uses the 'checkpoint_path' argument
    img_dir=eval_img_dir,
    gt_dir=eval_gt_dir,
    output_dir=output_dir,
    n_clicks=5,  # Default from eval.py, adjust if needed
    n_cases=num_eval_cases,
    n_classes_max=None,  # Default from eval.py
    use_wandb=True,  # Critical: eval.py will log its detailed metrics to the same wandb run
    wandb_project=wandb_project_name, # Ensure consistency
    verbose=False,  # Default from eval.py, adjust if needed
    save_segs=False,  # Default from eval.py, consider False if space/time is an issue
    checkpoint_path=os.path.dirname(os.path.dirname(trained_chkpt_path)) # Pass the path two levels up from the finetuned checkpoint
)


wandb.finish() # End the wandb run