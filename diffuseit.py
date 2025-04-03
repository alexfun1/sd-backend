from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler
import torch
#import xformers
import os
from PIL import Image
import numpy as np
import random
import gc
import json

models_dir = os.getenv("MODELS_DIR", "/Users/alexeyfun-young/Downloads/")

def mem_flush():
    """Flush the GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    else:
        print("CUDA is not available. Cannot flush memory.")
    gc.collect()

def _random_seed():
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)

def get_default_loras():
    return ""


def get_model(model_name):
    """Read models from models.json and return config for the model by id."""
    models_file = os.path.join(os.path.dirname(__file__), "config/models.json")
    if not os.path.exists(models_file):
        raise FileNotFoundError(f"models.json not found at {models_file}")
    
    with open(models_file, "r") as f:
        models = json.load(f)
    
    for model in models["models"]:
        if model["id"] == model_name:
            break
    else:
        raise ValueError(f"Model '{model_name}' not found in models.json")
    return model

def txt2img(prompt, negative_prompt="", model_name="pocsd15", width=768, height=512, seed=_random_seed()):
    """Generate an image from a text prompt using Stable Diffusion."""
    model = get_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "mps"

    # Set random seed
    if seed is None:
        seed = _random_seed()
    generator = torch.manual_seed(seed)

    # Load the model
    match model["type"]:
        case "sd15":
            pipeline = StableDiffusionPipeline.from_single_file(os.path.join(models_dir, model["file"]), torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
        case "sdxl":
            pipeline = StableDiffusionXLPipeline.from_single_file(os.path.join(models_dir, model["file"]), torch_dtype=torch.float16, use_safetensors=True)
        case "pony":
            pipeline = StableDiffusionPipeline.from_single_file(os.path.join(models_dir, model["file"]), torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
        case "sd2":
            pipeline = StableDiffusionPipeline.from_single_file(os.path.join(models_dir, model["file"]), torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
    
    #settings and optimization of CPU/Memory usage
    pipeline.to(device)
    #pipeline.set_progress_bar_config(disable=True)
    if torch.cuda.is_available():
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_vae_slicing()
        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.enable_model_cpu_offload()

    # Scheduler
    match model["settings"]["scheduler"]:
        case "DDIM":
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        case "PNDM":
            pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
        case "LMSDiscreteScheduler":
            pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        case "DPMSolverMultistepScheduler":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++")
        case "DPMSolverSinglestepScheduler":
            pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config)
        case "Euler A":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        case "Euler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        case "Heun":
            pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
    
    # Generate the image
    image = pipeline(prompt, 
                     negative_prompt=negative_prompt, 
                     generator=generator, 
                     width=width, 
                     height=height,
                     guidance_scale=model["settings"]["guidance"],
                     num_inference_steps=model["settings"]["steps"],
                     clip_skip=model["settings"]["clip_skip"],
                     ).images[0]
    
    return image