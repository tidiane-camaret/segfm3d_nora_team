import os 
import torch

from src.methods.nninteractivecore import nnInteractiveCorePredictor
from src.config import config

verbose = True

predictor = nnInteractiveCorePredictor(
    checkpoint_path=os.path.join(
        config["NNINT_CKPT_DIR"], "nnInteractive_v1.0"
    ),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=verbose,
)

network = predictor.network

del predictor
"""
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_training_output_dir = "/work/dlclarge2/ndirt-SegFM3D/model_checkpoints/nnint/nnInteractive_v1.0"
dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
plans = load_json(os.path.join(model_training_output_dir, 'plans.json'))
plans_manager = PlansManager(plans)
checkpoint = torch.load(os.path.join(model_training_output_dir,"fold_0/checkpoint_final.pth"), map_location=device, weights_only=False)

configuration_name = checkpoint['init_args']['configuration']

parameters = checkpoint['network_weights']

configuration_manager = plans_manager.get_configuration(configuration_name)

trainer_name = checkpoint['trainer_name']

num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
trainer_class = recursive_find_python_class(join(nnInteractive.__path__[0], "trainer"),
                                            trainer_name, 'nnInteractive.trainer')
if trainer_class is None:
    print(f'Unable to locate trainer class {trainer_name} in nnInteractive.trainer. '
                        f'Please place it there (in any .py file)!')
    print('Attempting to use default nnInteractiveTrainer_stub. If you encounter errors, this is where you need to look!')
    trainer_class = nnInteractiveTrainer_stub

network = trainer_class.build_network_architecture(
    configuration_manager.network_arch_class_name,
    configuration_manager.network_arch_init_kwargs,
    configuration_manager.network_arch_init_kwargs_req_import,
    num_input_channels,
    plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
    enable_deep_supervision=False
).to(device)
network.load_state_dict(parameters)





model_info = {
    "model_type": type(network).__name__,
    "model_module": type(network).__module__,
    "num_parameters": sum(p.numel() for p in network.parameters()),
    "input_channels": num_input_channels,
    "output_classes": plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
    "configuration_name": configuration_name,
    "network_arch_class_name": configuration_manager.network_arch_class_name,
    "network_arch_init_kwargs": configuration_manager.network_arch_init_kwargs,
    "network_arch_init_kwargs_req_import": configuration_manager.network_arch_init_kwargs_req_import
}
"""
print("Model Info:")
for key, value in model_info.items():
    print(f"{key}: {value}")
print("Model loaded successfully.")