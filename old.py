def set_pipeline(model):
    """Set the pipeline for the model."""
    if model["type"] == "sd15":
        pipeline = StableDiffusionPipeline.from_single_file(h.get_model_file(model["file"]), torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
    elif model["type"] == "sdxl":
        pipeline = StableDiffusionXLPipeline.from_single_file(h.get_model_file(model["file"]), torch_dtype=torch.float16, use_safetensors=True)
    elif model["type"] == "pony":
        pipeline = StableDiffusionPipeline.from_single_file(h.get_model_file(model["file"]), torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
    elif model["type"] == "sd2":
        pipeline = StableDiffusionPipeline.from_single_file(h.get_model_file(model["file"]), torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
    else:
        raise ValueError(f"Unknown model type: {model['type']}")
    pipeline = set_scheduler(pipeline, model)
    pipeline = optimize_pipeline(pipeline)
    
    return pipeline



def optimize_pipeline(pipeline):
    if torch.cuda.is_available():
        try:
           pipeline.vae.to("cuda")
        except Exception as e:
           print(f"Error moving VAE to CUDA: {e}")
           return None
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_vae_slicing()
        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.enable_model_cpu_offload()
    return pipeline