import yaml
import socket
from pathlib import Path

def detect_machine():
    """Detect which machine we're on based on hostname"""
    hostname = socket.gethostname()
    
    if "loki" in hostname or "nero" in hostname:
        return "nora"
    elif "mlgpu" in hostname or "kis3bat" in hostname:
        return "meta"
    else:
        return "default"
    
def load_config(config_path=None, machine=None):
    """
    Load configuration
    
    Args:
        config_path: Path to config file (or None to use default)
        machine: Machine name to use (or None to auto-detect)
    """
    # Determine config path
    if config_path is None:
        config_path = Path(__file__).parent / "../config.yaml"
    
    # Load all configs
    with open(config_path, "r") as f:
        all_configs = yaml.safe_load(f)
    
    machine = detect_machine()
    
    # Fall back to default if needed
    if machine not in all_configs:
        print(f"Warning: No configuration found for machine '{machine}', using default")
        machine = "default"
    
    config = all_configs[machine]
    print(f"Loaded configuration for machine: {machine}")
    
    return config

# Create a singleton config instance
config = load_config()

