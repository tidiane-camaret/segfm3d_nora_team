import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.multi_channel_transform import AddPromptChannelsTransform
from typing import Union, Tuple, List

from torch import nn

class CustomTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Set number of epochs
        self.num_epochs = 1
        print(f"Initializing CustomTrainer with {self.num_input_channels} input channels")
        
        # Update the network configuration to use the new number of input channels
    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                              do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels,
                              regions, ignore_label):
        # Get the default transforms
        transforms = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label
        )
        
        # Add your custom transform at the beginning of the pipeline
        transforms.transforms.insert(0, AddPromptChannelsTransform())
        
        return transforms 

    def get_validation_transforms(self, deep_supervision_scales, is_cascaded, foreground_labels, regions, ignore_label):
        transforms = super().get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels, regions, ignore_label)
        transforms.transforms.insert(0, AddPromptChannelsTransform())
        return transforms
    
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        print(f"Building network architecture with {num_input_channels + 7} input channels")
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + 7,  # Add 7 channels (1 original + 6 empty)
            num_output_channels,
            enable_deep_supervision
        )