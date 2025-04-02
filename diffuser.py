from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import gc
import random
import helper as h
from io import BytesIO
import base64
import os

def flush():
  gc.collect()
  torch.cuda.empty_cache()

def txt2img(model, prompt, negative_prompt, width, height, num_images = 1):
  # Set the random seed for reproducibility
  generator = torch.manual_seed(random.randint(0, 2**32 - 1))
  
  # Load the model
  pipe = StableDiffusionXLPipeline.from_single_file(f"{h.models_dir}/{model}", use_safetensors=True, torch_dtype = torch.float16)
  pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
  pipe.enable_vae_slicing()
  try:
    pipe.vae.to("cuda")
  except Exception as e:
    print(f"Error moving VAE to CUDA: {e}")
    return None
  pipe.enable_model_cpu_offload()
  pipe.enable_attention_slicing()
  pipe.enable_xformers_memory_efficient_attention()
  pipe.enable_vae_slicing()

  # Generate images
  images = []
  for i in range(num_images):
          image = pipe(prompt=prompt,
                                   negative_prompt=negative_prompt,
                                   width=width,
                                   height=height,
                                   num_inference_steps=20,
                                   generator=generator).images[0]
          images.append(image)
  flush()
  return images
