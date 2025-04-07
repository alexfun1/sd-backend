from diffusers import StableDiffusionXLPipeline as sdxl
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler, 
    HeunDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    DPMSolverSinglestepScheduler,
    DDIMScheduler, 
    PNDMScheduler, 
    LMSDiscreteScheduler)
import torch
import helpers as h
import schedulers as s

def build_sdxl_pipeline(model):
    pipeline = sdxl.from_single_file(h.get_model_file(model["file"]), torch_dtype=torch.float16, use_safetensors=True)
    pipeline.scheduler = s.set_scheduler(pipeline.scheduler.config, model["settings"]["scheduler"], model["settings"]["use_karras"])
    return pipeline