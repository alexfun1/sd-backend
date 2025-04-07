#from diffusion import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
#from diffusion import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler
import torch
#import xformers
import os
from PIL import Image
import numpy as np
import random
import gc
import json
import helpers as h
from compel import Compel, ReturnedEmbeddingsType
import prompter as p
import diffusion as d

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
                                                                            
def txt2img(prompt, negative_prompt="", model_name="pocsd15", width=768, height=512, seed=-1):
    """Generate an image from a text prompt using Stable Diffusion."""
    model = get_model(model_name)
    print(f"Model: {model}")
    vram = h.get_gpu_memory()
    print(f"VRAM: {vram}")

    device = "cuda" if torch.cuda.is_available() else "mps"

    # Set random seed
    if seed == -1:
        seed = _random_seed()
    generator = torch.manual_seed(seed)

    pipeline = d.set_pipeline(model)
    
    if pipeline is None:
        print(f"Pipeline is None for model: {model}")
        return None

    conditioning, pooled = p.prompt_embedding(pipeline, model["type"], prompt, negative_prompt)
    if conditioning is None:
        print(f"Conditioning is None for model: {model_name}")
        return None
    
    #compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    #conditioning = compel(prompt)

    # Generate the image
    if model["type"].upper() == "SD15":
        image = pipeline(
                    #prompt, 
                    prompt_embeds=conditioning, 
                    generator=generator, 
                    width=width, 
                    height=height,
                    guidance_scale=model["settings"]["guidance"],
                    num_inference_steps=model["settings"]["steps"],
                    clip_skip=model["settings"]["clip_skip"],
                    ).images[0]
    elif model["type"].upper() == "SDXL":
        image = pipeline(
                    #prompt, 
                    prompt_embeds=conditioning[0:1], 
                    pooled_prompt_embeds=pooled[0:1], 
                    negative_prompt_embeds=conditioning[1:2], 
                    negative_pooled_prompt_embeds=pooled[1:2], 
                    generator=generator, 
                    width=width, 
                    height=height,
                    guidance_scale=model["settings"]["guidance"],
                    num_inference_steps=model["settings"]["steps"],
                    clip_skip=model["settings"]["clip_skip"],
                    ).images[0]
    max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
    print('Max. memory used:', max_memory, 'GB')
    mem_flush()
    return image

def model_tests(prompt="A beautiful landscape",negative_prompt=""):
    """Run tests on the models."""
    models = ["pocsd15", "pocsdxl"]
    schedulers = ["DDIM", "PNDM", "LMSDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverSinglestepScheduler", "Euler A", "Euler"]
    for model_name in models:
        for scheduler in schedulers:
            print(f"Testing model: {model_name} with scheduler: {scheduler}")
            model = get_model(model_name)
            model["settings"]["scheduler"] = scheduler
            pipeline = d.set_pipeline(model)
            if pipeline is None:
                print(f"Pipeline is None for model: {model_name} with scheduler: {scheduler}")
                continue
            try:
                image = txt2img(prompt=prompt, negative_prompt=negative_prompt, model_name=model_name, width=768, height=512, seed=-1)
                #image.show()
                image.save(f"{model_name}_{scheduler}_output.png")
            except Exception as e:
                print(f"Error generating image with {model_name} and scheduler {scheduler}: {e}")
                mem_flush()

if __name__ == "__main__":
    try:
        #image =txt2img("A beautiful landscape", model_name="pocsdxl", width=1024, height=1024, seed=-1)
        #image.show()
        #image.save("output.png")
        prompt="1girl, long hair, breasts, smile, open mouth, blue eyes, large breasts, blonde hair, thighhighs, medium breasts, standing, nipples, underwear, ass, hetero, nude, teeth, multiple boys, penis, pussy, tongue, solo focus, indoors, looking back, dark skin, 2boys, sex, grin, white thighhighs, bra, vaginal, see-through, lips, pubic hair, uncensored, kneeling, makeup, anus, bed, no panties, bottomless, erection, chair, female pubic hair, garter straps, testicles, dark-skinned male, table, 3boys, underwear only, curtains, sex from behind, bent over, lingerie, all fours, lipstick, group sex, anal, male pubic hair, lace trim, lace, handjob, white bra, hand on another's head, doggystyle, garter belt, threesome, large penis, interracial, hanging breasts, gangbang, mmf threesome, imminent penetration, bedroom, lace-trimmed legwear, double penetration, foreskin, grabbing another's hair, netorare, spitroast, cheating (relationship), lace bra"
        neg_prompt = "bad anatomy, bad hands, bad feet, ugly, blurry, out of focus, low quality, worst quality, lowres, normal quality, jpeg artifacts, signature, watermark, username, artist name"
        model_tests(
            prompt = prompt,
            negative_prompt = neg_prompt)
    except KeyboardInterrupt:
        # Handle graceful exit
        # Perform any necessary cleanup here
        mem_flush()
        print("\nApplication interrupted. Exiting gracefully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        mem_flush()
        print("\nExiting due to error.")