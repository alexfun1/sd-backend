from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset
from PIL import Image
import torch
import os, json
from accelerate import Accelerator
from torchvision import transforms

# 🧾 Dataset Class
class ImageTextDataset(Dataset):
    def __init__(self, image_folder, metadata_path, tokenizer, resolution=512):
        with open(metadata_path, 'r') as f:
            self.metadata = [json.loads(line) for line in f]
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image = Image.open(os.path.join(self.image_folder, item['file_name'])).convert("RGB")
        text = item["text"]
        return {
            "pixel_values": self.transform(image),
            "input_ids": self.tokenizer(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids[0]
        }

# 🏗 Load Base SD Model (use sdxl-base for SDXL)
model_id = "runwayml/stable-diffusion-v1-5"  # Or try "stabilityai/stable-diffusion-xl-base-1.0"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = StableDiffusionPipeline.from_pretrained(model_id).unet

# ⚙️ Setup LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none",
)

unet = get_peft_model(unet, lora_config)

# 🧪 Load Dataset
dataset = ImageTextDataset("datasets/ds-test/images", "datasets/ds-test/metadata.jsonl", tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# 🚀 Accelerator for training
accelerator = Accelerator()
unet, text_encoder, dataloader = accelerator.prepare(unet, text_encoder, dataloader)

optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

# 🔁 Training Loop (Simple Example)
unet.train()
for epoch in range(10):
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(accelerator.device)
        input_ids = batch["input_ids"].to(accelerator.device)

        # Noisy latents (skipped for brevity), compute loss from U-Net
        # In real code: encode with VAE, add noise, denoise with U-Net, calc loss

        # Dummy forward for example
        outputs = unet(pixel_values)
        loss = outputs[0].mean()

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} completed")

# 💾 Save LoRA weights
unet.save_pretrained("lora-trained")

