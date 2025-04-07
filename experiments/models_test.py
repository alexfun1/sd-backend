from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler, 
    HeunDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    DPMSolverSinglestepScheduler,
    DDIMScheduler, 
    PNDMScheduler, 
    LMSDiscreteScheduler)
from diffusers.utils import make_image_grid
import torch
import random
import gc
import os
import json

models_dir = os.getenv("MODELS_DIR", "/home/fun/AI-Platforms/stable-diffusion-webui-forge/models/Stable-diffusion")
loras_dir = os.getenv("LORAS_DIR", "/home/fun/AI-Platforms/stable-diffusion-webui-forge/models/Loras")
default_loras_xl = ["add-detail-xl.safetensors",
                    "Expressive_H-000001.safetensors",
                    "incase_style_v3-1_ponyxl_ilff.safetensors"
                   ]
default_loras_xl_scale=[0.7, 0.7, 0.7]
default_loras_sd = ["add-detail-xl.safetensors",
                    "incasestylez-v3.safetensors"
                   ]
default_loras_sd_scale=[0.7, 0.7]


def list_safetensors_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.safetensors')]

def load_default_loras(pipeline, type):
    if type == "sdxl":
        for lora in default_loras_xl:
            pipeline.load_lora_weights(os.path.join(loras_dir, lora), weight_name=lora)
            
    elif type == "sd15":
        for lora in default_loras_sd:
            pipeline.load_lora_weights(os.path.join(loras_dir, lora), weight_name=lora)
    else:
        raise ValueError(f"Unknown model type: {type}")
    #pipeline.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

def sd_pipeline(file):
    pipeline = StableDiffusionPipeline.from_single_file(
      f"/home/fun/AI-Platforms/stable-diffusion-webui-forge/models/Stable-diffusion/{file}",
      variant="fp16",
      use_safetensors=True,
      torch_dtype=torch.float16
    )
    
    pipeline.enable_attention_slicing()
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_sequential_cpu_offload()
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()
    return pipeline
# Model File settings:
# {
#   "id": "sd15",
#   "file": "sd15.safetensors",
#   "type": "sd15",
#   "settings": {
#     "scheduler": "DDIM",
#     "use_karras": true,
#     "width": 512,
#     "height": 512,
#     "seed": -1,
#     "num_inference_steps": 30,
#     "guidance": 7.5,
#     "clip_skip": 2
#}

def add_model_to_config(file, model_id, model_type, config_path="model_config.json"):

    # Load existing config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = []

    # Check if model already exists in config
    for model in config:
        if model["file"] == file:
            print(f"Model {file} already exists in the config.")
            return

    # Add new model entry
    new_model = {
        "id": model_id,
        "file": file,
        "type": model_type,
        "settings": {
            "scheduler": "DDIM",
            "use_karras": True,
            "width": 512,
            "height": 512,
            "seed": -1,
            "num_inference_steps": 30,
            "guidance": 7.5,
            "clip_skip": 2
        }
    }
    config.append(new_model)

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Model {file} added to config.")

def set_scheduler(pipeline, scheduler_type, use_karras):
    if scheduler_type == "DDIM":
        scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=use_karras)
    elif scheduler_type == "PNDM":
        scheduler = PNDMScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=use_karras)
    elif scheduler_type == "LMS":
        scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=use_karras)
    elif scheduler_type == "Heun":
        scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=use_karras)
    elif scheduler_type == "Euler A":
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=use_karras)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    pipeline.scheduler = scheduler
    return pipeline

def sdxl_pipeline(file):
    pipeline = StableDiffusionXLPipeline.from_single_file(
      f"/home/fun/AI-Platforms/stable-diffusion-webui-forge/models/Stable-diffusion/{file}",
      variant="fp16",
      use_safetensors=True,
      torch_dtype=torch.float16
    )
    
    pipeline.enable_attention_slicing()
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_sequential_cpu_offload()
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()
    return pipeline

def do_img(model, type, prompt, negative_prompt):
    prompt = ["1girl, solo, long hair, breasts, short hair, multiple mature women, curvy, chubby, white background, 1boy, gloves, navel, 2girls, jewelry, nipples, hetero, thighs, sweat, nude, small breasts, lying, penis, barefoot, pussy, solo focus, indoors, dark skin, sex, spread legs, stomach, cum, on back, twitter , mole, blurry, vaginal, flat chest, dark-skinned female, completely nude, pillow, pubic hair, uncensored, tattoo, blurry background, pov, pussy juice, bed, depth of field, cum in pussy, on bed, bed sheet, arm support, from above, erection, female pubic hair, dark-skinned male, table, cum on body, male pubic hair, close-up, veins, freckles, clothed female nude male, out of frame, missionary, wooden floor, veiny penis, interracial, clitoris, head out of frame, imminent penetration, pov crotch, imminent vaginal, lower body, mole on thigh, twitter , clitoral hood, just the tip, mole on stomach, guided penetration",
              "1girl, mature woman, curvy, chubby, long hair, looking at viewer, open mouth, blue eyes, simple background, shirt, blonde hair, brown hair, 1boy, bare shoulders, upper body, hetero, nude, penis, sleeveless, tongue, solo focus, cum, black eyes, from side, grey eyes, eyelashes, lips, uncensored, profile, makeup, saliva, blue background, erection, half-closed eyes, testicles, oral, portrait, looking up, fellatio, fishnets, close-up, eyeshadow, veins, empty eyes, cum in mouth, nose, red lips, dirty, mascara, asian, runny makeup", 
             ]
    schedulers = ["DPMS","DPMS+Karras","Euler A", "Euler A + Karras"]

    
    generator = [torch.Generator().manual_seed(random.randint(0, 2**32 - 1)) for _ in range(len(prompt))]
    if type == "sd15":
        pipeline = sd_pipeline(model)
        compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
        prompt_embeds = compel(prompt)
        for s in schedulers:
            match s:
                case "DPMS":
                    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                case "DPMS+Karras":
                    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
                case "Euler A":
                    scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
                case "Euler A + Karras":
                    scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
                case _:
                    raise ValueError(f"Unknown scheduler: {scheduler}")
            pipeline.scheduler = scheduler
            images = pipeline(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=30).images
            id = file.split(".")[0]
            for i, image in enumerate(images):
                image.save(f"output/{id}_{s}_{i}.png")
                #image.show()
    elif type == "sdxl":
        pipeline = sdxl_pipeline(model)
        compel = Compel(
                        tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
                        text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True]
                    )
        conditioning, pooled = compel(prompt)
        for s in schedulers:
            match s:
                case "DPMS":
                    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                case "DPMS+Karras":
                    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
                case "Euler A":
                    scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
                case "Euler A + Karras":
                    scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
                case _:
                    raise ValueError(f"Unknown scheduler: {scheduler}")
            pipeline.scheduler = scheduler
            images = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=30).images
            for i, image in enumerate(images):
                id = file.split(".")[0]
                image.save(f"output/{id}_{s}_{i}.png")
                #image.show()
    else:
        raise ValueError(f"Unknown model type: {type}")

    add_model_to_config(file, file.split(".")[0], "sdxl")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

safetensors_files = list_safetensors_files("/home/fun/AI-Platforms/stable-diffusion-webui-forge/models/Stable-diffusion")
print("Found .safetensors files:", safetensors_files)
neg_prompt = "bad anatomy, bad hands, bad feet, ugly, blurry, out of focus, low quality, worst quality, lowres, normal quality, jpeg artifacts, signature, watermark, username, artist name"

for file in safetensors_files:
    print(f"Testing {file}...")
    try:
        do_img(file, "sdxl", None, neg_prompt)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        print("Trying SDXL...")
        try:
            do_img(file, "sd15", None, neg_prompt)
        except Exception as e:
            print(f"Error processing {file} with SDXL: {e}")
            continue
    finally:
        torch.cuda.empty_cache()
        gc.collect()