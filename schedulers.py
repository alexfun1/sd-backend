from diffusers import (
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler, 
    HeunDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    DPMSolverSinglestepScheduler,
    DDIMScheduler, 
    PNDMScheduler, 
    LMSDiscreteScheduler)

def set_scheduler(config, scheduler, karras=False):
    """Set the scheduler for the pipeline."""
    match scheduler:
        case "DDIM":
            result = DDIMScheduler.from_config(config)
        case "PNDM":
            result = PNDMScheduler.from_config(config)
        case "LMSDiscreteScheduler":
            result = LMSDiscreteScheduler.from_config(config)
        case "DPMSolverMultistepScheduler":
            if karras: 
               result = DPMSolverMultistepScheduler.from_config(config, algorithm_type="sde-dpmsolver++",use_karras_sigmas=True)
            else:
                result = DPMSolverMultistepScheduler.from_config(config, algorithm_type="sde-dpmsolver++")
        case "DPMSolverSinglestepScheduler":
            result = DPMSolverSinglestepScheduler.from_config(config)
        case "Euler A":
            result = EulerAncestralDiscreteScheduler.from_config(config)
        case "Euler":
            result = EulerDiscreteScheduler.from_config(config)
        case "Heun":
            result = HeunDiscreteScheduler.from_config(config)
        case _:
            result = DPMSolverMultistepScheduler.from_config(config)
    return result