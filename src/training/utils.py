from contextlib import contextmanager
import signal
import threading
import torch
import numpy as np
import hashlib

@contextmanager
def defer_keysignal():
    signum = signal.SIGINT
    # Based on https://stackoverflow.com/a/71330357/1319998

    original_handler = None
    defer_handle_args = None

    def defer_handle(*args):
        nonlocal defer_handle_args
        defer_handle_args = args

    # Do nothing if
    # - we don't have a registered handler in Python to defer
    # - or the handler is not callable, so either SIG_DFL where the system
    #   takes some default action, or SIG_IGN to ignore the signal
    # - or we're not in the main thread that doesn't get signals anyway
    original_handler = signal.getsignal(signum)
    if (
        original_handler is None
        or not callable(original_handler)
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    try:
        signal.signal(signum, defer_handle)
        yield
    finally:
        signal.signal(signum, original_handler)
        if defer_handle_args is not None:
            original_handler(*defer_handle_args)


@contextmanager
def defer_keysignal_with_grad():
    # Wrap the existing defer_keysignal context manager
    with defer_keysignal():
        # Turn on PyTorch gradient mode
        torch.set_grad_enabled(True)
        try:
            yield
        finally:
            # Ensure gradient mode is turned off at the end
            torch.set_grad_enabled(False)


def compute_binary_dsc(pred, mask):
    intersect = np.sum(pred * mask)
    return 2 * intersect / (np.sum(pred) + np.sum(mask))



def fast_id_to_deterministic_float(id):
    hash_digest = hashlib.sha256(str(id).encode("utf-8")).digest()
    seed = int.from_bytes(hash_digest[:4], "big", signed=False)
    float_val = float(seed) / 4294967296.0
    return float_val


fast_ids_to_deterministic_floats = np.vectorize(fast_id_to_deterministic_float)


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

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    ContrastTransform,
    BGContrast,
)
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

test_transform = ComposeTransforms(
    [
        NormalizeSingleImageTransform(),
        MONAIFixedSpatialTransform(),
        RemoveLabelTansform(segmentation_channels=None, label_value=-1, set_to=0),
        AddBBoxAndEmptyChannelsSingleClassTransform(only_2d_bbox=only_2d_bbox),
        AddSegToImageTransform(),
    ]
)

# let's make very little augmentation
train_transform = ComposeTransforms(
    [
        NormalizeSingleImageTransform(),
        MONAIFixedSpatialTransform(),
        # RandomTransform(
        #     apply_probability=0.1,
        #     transform=GaussianNoiseTransform(
        #         noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True
        #     ),
        # ),
        # RandomTransform(
        #     apply_probability=0.2,
        #     transform=GaussianBlurTransform(
        #         blur_sigma=(0.5, 1.0),
        #         benchmark=True,
        #         synchronize_channels=False,
        #         synchronize_axes=False,
        #         p_per_channel=0.5,
        #     ),
        # ),
        # MirrorTransform(allowed_axes=(0, 1, 2)),
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