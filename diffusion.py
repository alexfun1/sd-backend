import schedulers as s
import helpers as h
import sd15 as sd
import sdxl

def set_pipeline(model):
    print(f"Setting pipeline for model type: {model['type']}")
    """Set the pipeline for the model."""
    if model["type"] == "sd15":
        pipeline = sd.build_sd15_pipeline(model)
    elif model["type"] == "sdxl":
        pipeline = sdxl.build_sdxl_pipeline(model)
    elif model["type"] == "pony":
        pipeline = sd.build_sd15_pipeline(model)
    elif model["type"] == "sd2":
        pipeline = sd.build_sd15_pipeline(model)
    else:
        raise ValueError(f"Unknown model type: {model['type']}")
    #pipeline = optimize_pipeline(pipeline)
    
    return pipeline