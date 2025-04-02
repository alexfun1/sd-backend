import os

models_dir = "/home/fun/AI-Platforms/stable-diffusion-webui-forge/models/Stable-diffusion"

def load_models_from_dir():
    """
    Load all model files from the specified directory.
    
    Args:
        model_dir (str): Directory containing model files.
    
    Returns:
        list: List of model file paths.
    """
    res = []
    # Get a list of all files in the directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".safetensors")]
    
    # Print available model files
    print("Available model files:")
    for idx, file in enumerate(model_files):
        print(f"{idx}: {file}")
        res.append(file)
    
    return res