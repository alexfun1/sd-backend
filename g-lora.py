import os
import argparse
import torch
from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    )
from diffusers import DDPMScheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import json
import gc

class MetadataDataset(Dataset):
    def __init__(self, image_folder, metadata_file, tokenizer, size=512, center_crop=False):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.metadata = self._load_metadata(metadata_file)
        self.image_files = list(self.metadata.keys())
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.to_tensor = transforms.ToTensor()

    def _load_metadata(self, metadata_file):
        metadata = {}
        with open(metadata_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    metadata[data["file_name"].split("/")[-1]] = data["text"]
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {line.strip()}. Error: {e}")
        return metadata

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        caption = self.metadata[image_file]
        img = Image.open(image_path).convert("RGB")
        img = self.image_transforms(img) # the transforms now include to_tensor()
        caption_ids = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return {"instance_images": img, "instance_prompt_ids": caption_ids[0]}
def train_lora(image_folder, metadata_file, pretrained_model_name_or_path, output_dir, train_batch_size=1, gradient_accumulation_steps=1, learning_rate=1e-4, max_train_steps=800, mixed_precision="no", seed=42, lora_rank=8, checkpointing_steps=500):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision)
    print(accelerator.state, flush=True)
    if seed:
        torch.manual_seed(seed)

    if "xl" in pretrained_model_name_or_path.lower():
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
        tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler") #load the scheduler
    else:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler") #load the scheduler


    unet.requires_grad_(False)
    unet_lora_config = torch.nn.ModuleList([])
    for name, module in unet.named_modules():
        if "attn2" in name or "to_k" in name or "to_q" in name or "to_v" in name or "to_out.0" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear):
                    unet_lora_config.append(torch.nn.Linear(child_module.in_features, child_module.out_features, bias=False))
                    unet_lora_config[-1].lora_A = torch.nn.Parameter(torch.randn(child_module.in_features, lora_rank))
                    unet_lora_config[-1].lora_B = torch.nn.Parameter(torch.zeros(lora_rank, child_module.out_features))
                    child_module.weight = unet_lora_config[-1].weight
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)
    if "xl" in pretrained_model_name_or_path.lower():
        text_encoder_2.to(accelerator.device)

    train_dataset = MetadataDataset(image_folder=image_folder, metadata_file=metadata_file, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0) #num_workers=0 for memory

    optimizer = torch.optim.AdamW(unet_lora_config.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    unet_lora_config, optimizer, train_dataloader = accelerator.prepare(unet_lora_config, optimizer, train_dataloader)
    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = (max_train_steps * gradient_accumulation_steps) // num_update_steps_per_epoch
    total_train_steps = num_train_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(total_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            latents = torch.randn((batch["instance_images"].shape[0], 4, 64, 64), device=accelerator.device)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=accelerator.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #use the loaded scheduler.
            encoder_hidden_states = text_encoder(batch["instance_prompt_ids"])[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        logs = {"loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)

        if (step + 1) % checkpointing_steps == 0:
            if accelerator.is_main_process:
                if "xl" in pretrained_model_name_or_path.lower():
                    pipeline = StableDiffusionXLPipeline(text_encoder=accelerator.unwrap_model(text_encoder), text_encoder_2=accelerator.unwrap_model(text_encoder_2), vae=StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path).vae, unet=accelerator.unwrap_model(unet), tokenizer=tokenizer, tokenizer_2=tokenizer_2, scheduler=accelerator.noise_scheduler, safety_checker=None, feature_extractor=None)
                else:
                    pipeline = StableDiffusionLoraPipeline(text_encoder=accelerator.unwrap_model(text_encoder), vae=StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).vae, unet=accelerator.unwrap_model(unet), tokenizer=tokenizer, scheduler=accelerator.noise_scheduler, safety_checker=None, feature_extractor=None)
                pipeline.save_pretrained(output_dir)
        gc.collect() #memory saving
        torch.cuda.empty_cache() #memory saving

    if accelerator.is_main_process:
        if "xl" in pretrained_model_name_or_path.lower():
            pipeline = StableDiffusionXLPipeline(text_encoder=accelerator.unwrap_model(text_encoder), text_encoder_2=accelerator.unwrap_model(text_encoder_2), vae=StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path).vae, unet=accelerator.unwrap_model(unet), tokenizer=tokenizer, tokenizer_2=tokenizer_2, scheduler=accelerator.noise_scheduler, safety_checker=None, feature_extractor=None)
        else:
            pipeline = StableDiffusionLoraPipeline(text_encoder=accelerator.unwrap_model(text_encoder), vae=StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).vae, unet=accelerator.unwrap_model(unet), tokenizer=tokenizer, scheduler=accelerator.noise_scheduler, safety_checker=None, feature_extractor=None)
        pipeline.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="sd-lora-model")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    args = parser.parse_args()
    train_lora(image_folder=args.image_folder, metadata_file=args.metadata_file, pretrained_model_name_or_path=args.pretrained_model_name_or_path, output_dir=args.output_dir, train_batch_size=args.train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, max_train_steps=args.max_train_steps, mixed_precision=args.mixed_precision, seed=args.seed, lora_rank=args.lora_rank, checkpointing_steps=args.checkpointing_steps)