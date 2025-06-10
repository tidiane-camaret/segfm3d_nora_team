import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import os
from src.config import config
import subprocess
import json


dataset_name = "Dataset010_ALL"
network_config_name = "CustomTrainer__nnUNetResEncUNetLPlans__3d_fullres_ps192"
nb_cases_eval = 10
os.environ["nnUNet_raw"] = os.path.join(config["DATA_DIR"], "nnunet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(config["DATA_DIR"], "nnUNet_preprocessed") 
os.environ["nnUNet_results"] = os.path.join(config["DATA_DIR"], "nnUNet_results")
os.environ["NNINT_CKPT_DIR"] = config["NNINT_CKPT_DIR"]

# Evaluate the 1.0 weights on the competition task
#command = f"python scripts/eval.py -ca {nb_cases_eval} -m nnint_custom --checkpoint_path {os.path.join(os.environ["NNINT_CKPT_DIR"] ,"nnInteractive_v1.0")}"
#os.system(command)




dataset_folder = os.path.join(os.environ["nnUNet_preprocessed"], dataset_name)
plan_json = json.load(open(os.path.join(dataset_folder, "nnUNetResEncUNetLPlans.json"), "r"))
dataset_json = json.load(open(os.path.join(dataset_folder, "dataset.json"), "r"))



fold = 0
#os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile for debugging
from nnunetv2.training.nnUNetTrainer.CustomTrainer import CustomTrainer

trainer = CustomTrainer(
    plans=plan_json,
    configuration="3d_fullres_ps192",
    fold=fold,
    dataset_json=dataset_json,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #freeze_decoder=False,
)
# Set the initial checkpoint path 
trainer.set_initial_checkpoint(os.path.join(os.environ["NNINT_CKPT_DIR"] ,"nnInteractive_v1.0/fold_0/checkpoint_final.pth"))
trainer.initialize()


trainer.run_training()

### Evaluate the trained model on the competition task

command = f"python scripts/eval.py -ca {nb_cases_eval} -m nnint_custom --checkpoint_path {os.environ['nnUNet_results']}/{dataset_name}/{network_config_name}"
print(command)
os.system(command)