import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from src.training.transforms import (
    AddBBoxAndEmptyChannelsTransform,
    AddSegToImageTransform,
    NormalizeSingleImageTransform,
)
from src.training.npzdataset import NPZDataset
from typing import Union, Tuple, List
import numpy as np
from torch import nn
import hashlib
import numpy as np
import os
from src.training.model_wrap import ModelPrevSegAndClickWrapper


class CustomTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Modify the number of input channels to account for the additional channels

        # Set number of epochs
        self.num_epochs = 1
        self.initial_lr = 3e-4
        self.weight_decay = 1e-5
        """
        # Update the network configuration to use the new number of input channels
        if hasattr(self, 'network'):
            self.network.conv_blocks_context[0].convs[0].conv.in_channels = self.num_input_channels
            self.network.conv_blocks_context[0].convs[0].all_modules[0].in_channels = self.num_input_channels
        """
        print("whatever custom trainer 2")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,)
        # constant LR
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1)
        return optimizer, lr_scheduler
    

    def get_tr_and_val_datasets(self):
        npz_folder = "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/SegFM3D/data/3D_train_npz_random_10percent_16G/CT/CT_AMOS"
        save_folder  = '/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/SegFM3D/data/own-preproc/CT/CT_AMOS'
        npz_files = np.array(sorted([f for f in os.listdir(npz_folder) if f.endswith(".npz")]))
        train_mask = fast_ids_to_deterministic_floats(npz_files) < 0.8
        train_files = npz_files[train_mask]
        test_files = npz_files[~train_mask]
        dataset_tr = NPZDataset(npz_folder, save_folder, train_files)
        dataset_val = NPZDataset(npz_folder, save_folder, test_files)
        return dataset_tr, dataset_val

    def get_training_transforms(
        self,
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        use_mask_for_norm,
        is_cascaded,
        foreground_labels,
        regions,
        ignore_label,
    ):
        # Get the default transforms
        transforms = super().get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        # Add your custom transform at the beginning of the pipeline
        transforms.transforms.insert(0, NormalizeSingleImageTransform())
        transforms.transforms.insert(1, AddBBoxAndEmptyChannelsTransform())
        transforms.transforms.append(AddSegToImageTransform())
        return transforms

    def get_validation_transforms(
        self,
        deep_supervision_scales,
        is_cascaded,
        foreground_labels,
        regions,
        ignore_label,
    ):
        transforms = super().get_validation_transforms(
            deep_supervision_scales,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        transforms.transforms.insert(0, NormalizeSingleImageTransform())
        transforms.transforms.insert(1, AddBBoxAndEmptyChannelsTransform())
        transforms.transforms.append(AddSegToImageTransform())
        return transforms

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + 7,
            2,  # nnunet handles one class segmentation still as CE so we need 2 outputs.
            enable_deep_supervision,
        )

        print("wrapping that network!", network.__class__)

        return ModelPrevSegAndClickWrapper(network)
