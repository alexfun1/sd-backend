import subprocess as sp
import os
import json
import random

models_dir = os.getenv("MODELS_DIR", "/Users/alexeyfun-young/Downloads/")
if not models_dir:
    raise ValueError("MODELS_DIR environment variable is not set.")
if not os.path.exists(models_dir):
    raise FileNotFoundError(f"Models directory not found at {models_dir}")

def get_model_file(file):
    return os.path.join(models_dir, f"{file}")

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def set_orientation(orientation, width, height):
    """Set the orientation of the image."""
    if orientation == "Landscape" and width > height:
        return width, height
    else:
        return height, width

def get_models_list():
    """Get the list of models."""
    models_file = os.path.join(os.path.dirname(__file__), "config/models.json")
    if not os.path.exists(models_file):
        raise FileNotFoundError(f"models.json not found at {models_file}")
    
    with open(models_file, "r") as f:
        models = json.load(f)
    
    res = []
    for model in models["models"]:
        res.append(model["id"])
    print(f"Models: {res}")
    return res