from diffusers import StableDiffusionPipeline as sd
from diffusers import AutoencoderKL
import helpers as h
import schedulers as s
import torch

#vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def build_sd15_pipeline(model):
    pipeline = sd.from_single_file(h.get_model_file(model["file"]), torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
    pipeline.scheduler = s.set_scheduler(pipeline.scheduler.config, model["settings"]["scheduler"], model["settings"]["use_karras"])
    mem = h.get_gpu_memory()
    #memory and cpu/gpu offload depending on type and memory available
    if mem[0] < 10000 and torch.cuda.is_available():
        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_vae_slicing()
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_vae_slicing()
        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.to(device)
    
    #if torch.cuda.is_available():
    #    try:
    #       pipeline.vae.to(device)
    #       pipeline.unet.to(device)
    #       pipeline.text_encoder.to(device)
    #       #pipeline.tokenizer.to(device)
    #       #pipeline.scheduler.to(device)
    #       return pipeline
    #    except Exception as e:
    #       print(f"Error moving components to CUDA: {e}")
    #       return None
    #else:
    return pipeline