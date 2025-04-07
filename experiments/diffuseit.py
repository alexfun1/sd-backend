from diffusion import StableDiffusionPipeline
import torch
import os
from PIL import Image
import numpy as np
import random

def set_some_env(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers"
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_CACHE_PATH"] = "/tmp/cuda"
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["CUDA_CACHE_MAXSIZE"] = "0"

def get_rand_seed():
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)
def get_model_list():
    """Get the list of available models."""
    return [
        "CompVis/stable-diffusion-v1-4",
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-base",
        "stabilityai/stable-diffusion-2-1-base",
        "stabilityai/stable-diffusion-2-1-finetuned",
        "stabilityai/stable-diffusion-2-inpainting",
        "stabilityai/stable-diffusion-2-depth",
        "stabilityai/stable-diffusion-2-colorization",
        "stabilityai/stable-diffusion-2-cartoonization",
        "stabilityai/stable-diffusion-2-sketch"
    ]
def get_model_info(model_name):
    """Get the information about the model."""
    model_info = {
        "CompVis/stable-diffusion-v1-4": "Stable Diffusion v1.4",
        "runwayml/stable-diffusion-v1-5": "Stable Diffusion v1.5",
        "stabilityai/stable-diffusion-2-base": "Stable Diffusion 2.0 Base",
        "stabilityai/stable-diffusion-2-1-base": "Stable Diffusion 2.1 Base",
        "stabilityai/stable-diffusion-2-1-finetuned": "Stable Diffusion 2.1 Finetuned",
        "stabilityai/stable-diffusion-2-inpainting": "Stable Diffusion 2.0 Inpainting",
        "stabilityai/stable-diffusion-2-depth": "Stable Diffusion 2.0 Depth",
        "stabilityai/stable-diffusion-2-colorization": "Stable Diffusion 2.0 Colorization",
        "stabilityai/stable-diffusion-2-cartoonization": "Stable Diffusion 2.0 Cartoonization",
        "stabilityai/stable-diffusion-2-sketch": "Stable Diffusion 2.0 Sketch"
    }
    return model_info.get(model_name, "")
def get_model_description(model_name):
    """Get the description of the model."""
    model_description = {
        "CompVis/stable-diffusion-v1-4": "Stable Diffusion v1.4 is a latent text-to-image diffusion model.",
        "runwayml/stable-diffusion-v1-5": "Stable Diffusion v1.5 is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-base": "Stable Diffusion 2.0 Base is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-1-base": "Stable Diffusion 2.1 Base is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-1-finetuned": "Stable Diffusion 2.1 Finetuned is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-inpainting": "Stable Diffusion 2.0 Inpainting is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-depth": "Stable Diffusion 2.0 Depth is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-colorization": "Stable Diffusion 2.0 Colorization is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-cartoonization": "Stable Diffusion 2.0 Cartoonization is a latent text-to-image diffusion model.",
        "stabilityai/stable-diffusion-2-sketch": "Stable Diffusion 2.0 Sketch is a latent text-to-image diffusion model."
    }
    return model_description.get(model_name, "")

def txt2img(prompt, negative_prompt, model_name, seed, width=512, height=512):
    """Generate an image from a text prompt using Stable Diffusion."""
    set_some_env(seed)
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Generate the image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height).images[0]
    
    return image
def img2img(prompt, negative_prompt, model_name, seed, init_image, strength=0.75):
    """Generate an image from a text prompt using Stable Diffusion."""
    set_some_env(seed)
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Preprocess the input image
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate the image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, init_image=init_image, strength=strength).images[0]
    
    return image
def inpaint(prompt, negative_prompt, model_name, seed, init_image, mask_image, strength=0.75):
    """Generate an image from a text prompt using Stable Diffusion."""
    set_some_env(seed)
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Preprocess the input image
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Preprocess the mask image
    mask_image = Image.open(mask_image).convert("L")
    mask_image = mask_image.resize((512, 512))
    
    # Generate the image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, init_image=init_image, mask_image=mask_image, strength=strength).images[0]
    
    return image
def upscale(prompt, negative_prompt, model_name, seed, init_image, scale=2):
    """Generate an image from a text prompt using Stable Diffusion."""
    set_some_env(seed)
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Preprocess the input image
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate the image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, init_image=init_image, scale=scale).images[0]
    
    return image
def colorize(prompt, negative_prompt, model_name, seed, init_image):
    """Generate an image from a text prompt using Stable Diffusion."""
    set_some_env(seed)
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Preprocess the input image
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate the image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, init_image=init_image).images[0]
    
    return image
def cartoonize(prompt, negative_prompt, model_name, seed, init_image):
    """Generate an image from a text prompt using Stable Diffusion."""
    set_some_env(seed)
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Preprocess the input image
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate the image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, init_image=init_image).images[0]
    
    return image
def sketch(prompt, negative_prompt, model_name, seed, init_image):
    """Generate an image from a text prompt using Stable Diffusion."""
    set_some_env(seed)
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Preprocess the input image
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate the image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, init_image=init_image).images[0]
    
    return image