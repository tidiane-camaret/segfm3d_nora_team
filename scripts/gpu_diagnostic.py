""" 
Test if GPU is available and print the result.

"""
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("GPU current device:", torch.cuda.current_device() if torch.cuda.is_available() else "No GPU")
print("GPU current device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No GPU")
print("GPU current device index:", torch.cuda.current_device() if torch.cuda.is_available() else "No GPU")
print("GPU current device properties:", torch.cuda.get_device_properties(torch.cuda.current_device()) if torch.cuda.is_available() else "No GPU")
print("GPU current device total memory:", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory if torch.cuda.is_available() else "No GPU")
print("GPU current device total memory (GB):", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3) if torch.cuda.is_available() else "No GPU")