import os, json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator
from torchvision import transforms
from peft import get_peft_model, LoraConfig
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                       AutoencoderKL, DDPMScheduler)
from transformers import AutoTokenizer

###################################
# Minimal local dataset class
###################################
class ImageDataset(Dataset):
    def __init__(self, img_folder, metadata_path, resolution=512):
        with open(metadata_path, 'r') as f:
            self.meta = [json.loads(line.strip()) for line in f]
        self.img_folder = img_folder
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        record = self.meta[idx]
        img_path = os.path.join(self.img_folder, record["file_name"])
        image = Image.open(img_path).convert("RGB")
        text = record["text"]
        return {
            "pixel_values": self.transform(image),
            "text": text
        }
    
def enable_lora_for_module(module, module_type="unet"):
    from peft import get_peft_model, LoraConfig

    # Different defaults depending on module_type
    if module_type == "unet":
        targets = ["to_q", "to_k", "to_v", "to_out.0"]
    elif module_type == "text_encoder_15":  # CLIP
        targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
    elif module_type == "text_encoder_xl":  # T5
        targets = ["q", "k", "v", "o"]  # or "q_proj","k_proj","v_proj","o_proj"
    else:
        raise ValueError("Unknown module type")

    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=targets,
        lora_dropout=0.1,
        bias="none"
    )

    # Wrap in PEFT
    lora_model = get_peft_model(module, config)

    # Freeze everything
    for _, param in lora_model.named_parameters():
        param.requires_grad = False

    # Enable just LoRA
    for name, param in lora_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    return lora_model

def enable_lora_for_text_encoder_15(text_encoder):
    # For SD1.5 text encoder (CLIP architecture)
    from peft import get_peft_model, LoraConfig

    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","out_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    # Wrap in PEFT
    lora_encoder = get_peft_model(text_encoder, config)

    # Freeze entire model
    for p in lora_encoder.parameters():
        p.requires_grad = False

    # Enable only LoRA submodules
    for name, p in lora_encoder.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True

    return lora_encoder

###################################
# Main training function
###################################
def train_lora(
    base_model_id: str,
    data_folder: str,
    output_lora_path: str = "lora-trained",
    resolution: int = 512,
    lr: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 1,
    train_unet: bool = True,
    train_text_encoder: bool = False
):
    """
    Trains a LoRA for either SD1.5 or SDXL, using a local folder with images + metadata.jsonl.
    Allows control over whether to train LoRA on the U-Net, text encoder, or both.

    Args:
        base_model_id (str):
            e.g. "runwayml/stable-diffusion-v1-5" or "stabilityai/stable-diffusion-xl-base-1.0"
        data_folder (str):
            Local dataset folder. Must contain an "images/" subfolder + "metadata.jsonl".
        output_lora_path (str):
            Where to save the LoRA weights.
        resolution (int):
            Image resolution for training. 512 for SD1.5, 1024 for SDXL, etc.
        lr (float):
            Learning rate for the AdamW optimizer.
        epochs (int):
            Number of training epochs to run.
        batch_size (int):
            Batch size per training step.
        train_unet (bool):
            If True, apply LoRA + train the U-Net.
        train_text_encoder (bool):
            If True, apply LoRA + train the text encoder (for SD1.5 = single text encoder; for SDXL = text_encoder_2).
    """
    # ---- Detect if it's SD1.5 or SDXL
    is_sdxl = "xl" in base_model_id.lower()

    # ---- Load base pipeline
    if is_sdxl:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    unet = pipe.unet
    vae = pipe.vae
    noise_scheduler = pipe.scheduler

    # For SD1.5, we have pipe.text_encoder, pipe.tokenizer
    # For SDXL, we have pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
    if is_sdxl:
        text_encoder_1 = pipe.text_encoder       # CLIP-based
        text_encoder_2 = pipe.text_encoder_2     # T5-based
        tokenizer_1 = pipe.tokenizer
        tokenizer_2 = pipe.tokenizer_2
    else:
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer

    # ---- Freeze everything by default
    for param in vae.parameters():
        param.requires_grad = False
    for param in unet.parameters():
        param.requires_grad = False

    if is_sdxl:
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)

    # ---- Setup LoRA for whichever modules we want to train
    # We'll do one config for each module that we want to tune.
    # def enable_lora_for_module(module):
    #     config = LoraConfig(
    #         r=4,
    #         lora_alpha=16,
    #         target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    #         lora_dropout=0.1,
    #         bias="none"
    #     )
    #     return get_peft_model(module, config)

    # If the user wants to train the U-Net, rewrap it in LoRA:
    if train_unet:
        unet = enable_lora_for_module(unet, module_type="unet")
        #for p in unet.trainable_parameters():
        #    p.requires_grad = True

    # If the user wants to train text encoder LoRA:
    # - For SD1.5: we just do text_encoder
    # - For SDXL: let's pick text_encoder_2 (T5) for simplicity
    if train_text_encoder:
        if is_sdxl:
            text_encoder_2 = enable_lora_for_module(text_encoder_2, module_type="text_encoder_xl")
            for p in text_encoder_2.trainable_parameters():
                p.requires_grad = True
        else:
            text_encoder = enable_lora_for_text_encoder_15(text_encoder)
            for p in text_encoder.trainable_parameters():
                p.requires_grad = True

    # ---- Load dataset
    dataset = ImageDataset(
        os.path.join(data_folder, "images"),
        os.path.join(data_folder, "metadata.jsonl"),
        resolution=resolution
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---- Setup optimizer & accelerator
    accelerator = Accelerator()
    if is_sdxl:
        # We'll wrap them in a list for accelerate
        modules_for_accelerator = [unet] if train_unet else []
        if train_text_encoder:
            modules_for_accelerator.append(text_encoder_2)
        modules_for_accelerator.append(dataloader)
        prepared = accelerator.prepare(*modules_for_accelerator)
        # Now we parse them out
        if train_unet and train_text_encoder:
            unet, text_encoder_2, dataloader = prepared
        elif train_unet and not train_text_encoder:
            unet, dataloader = prepared
        elif (not train_unet) and train_text_encoder:
            text_encoder_2, dataloader = prepared
        else:
            dataloader = prepared
    else:
        # SD1.5
        modules_for_accelerator = [unet] if train_unet else []
        if train_text_encoder:
            modules_for_accelerator.append(text_encoder)
        modules_for_accelerator.append(dataloader)
        prepared = accelerator.prepare(*modules_for_accelerator)
        if train_unet and train_text_encoder:
            unet, text_encoder, dataloader = prepared
        elif train_unet and not train_text_encoder:
            unet, dataloader = prepared
        elif (not train_unet) and train_text_encoder:
            text_encoder, dataloader = prepared
        else:
            dataloader = prepared

    # Collect parameters for optimizer
    params_to_optimize = []
    if train_unet:
        params_to_optimize += list(unet.parameters())
    if train_text_encoder:
        if is_sdxl:
            params_to_optimize += list(text_encoder_2.parameters())
        else:
            params_to_optimize += list(text_encoder.parameters())

    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)

    # ---- Training loop
    device = accelerator.device
    for epoch_i in range(epochs):
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)

            # encode images -> latents
            # For SD1.5, scale is 0.18215; for SDXL, it's 0.13025
            scale = 0.13025 if is_sdxl else 0.18215
            latents = vae.encode(pixel_values).latent_dist.sample() * scale

            # Add random noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode text
            texts = batch["text"]
            if is_sdxl:
                # We'll do minimal approach: text_encoder_2 for cross-attn
                # (since T5 is the usual cross-attn in official SDXL)
                # If user didn't train it, it's still loaded from base pipeline
                token_out = tokenizer_2(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).input_ids.to(device)
                text_hidden_states = text_encoder_2(token_out)[0]
            else:
                token_out = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).input_ids.to(device)
                text_hidden_states = text_encoder(token_out)[0]

            # U-Net forward
            # if user turned off unet training, it's still in no_grad, but let's call it
            model_pred = unet(
                noisy_latents,
                timesteps,
                text_hidden_states
            ).sample

            # MSE loss vs. actual noise
            loss = torch.nn.functional.mse_loss(model_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"[Epoch {epoch_i} | Step {step}] loss={loss.item():.4f}")

    # ---- Save LoRA weights
    if train_unet:
        unet.save_pretrained(output_lora_path + "_unet")
    if train_text_encoder:
        if is_sdxl:
            text_encoder_2.save_pretrained(output_lora_path + "_textenc")
        else:
            text_encoder.save_pretrained(output_lora_path + "_textenc")

    print(f"Done! LoRA weights saved under: {output_lora_path}")
    return output_lora_path
