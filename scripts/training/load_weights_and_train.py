import os
from src.config import config
os.environ["nnUNet_raw"] = os.path.join(config["DATA_DIR"], "nnunet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(config["DATA_DIR"], "nnUNet_preprocessed") 
os.environ["nnUNet_results"] = os.path.join(config["DATA_DIR"], "nnUNet_results")
os.environ["NNINT_CKPT_DIR"] = config["NNINT_CKPT_DIR"]


import json
dataset_folder = os.environ["nnUNet_preprocessed"]+ "/Dataset007_AMOS/"
plan_json = json.load(open(os.path.join(dataset_folder, "nnUNetResEncUNetLPlans.json"), "r"))
dataset_json = json.load(open(os.path.join(dataset_folder, "dataset.json"), "r"))


import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from nnunetv2.training.nnUNetTrainer.CustomTrainer import CustomTrainer
fold = 0
#os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile for debugging

trainer = CustomTrainer(
    plans=plan_json,
    configuration="3d_fullres_ps192",
    fold=fold,
    dataset_json=dataset_json,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
# Set the initial checkpoint path using the new setter method
trainer.set_initial_checkpoint(os.path.join(os.environ["NNINT_CKPT_DIR"] ,"nnInteractive_v1.0/fold_0/checkpoint_final.pth"))
trainer.initialize()


trainer.run_training()