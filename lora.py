from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
import torch, os, json
from PIL import Image
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# === Configs ===
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dataset_path = "datasets/ds-test"
resolution = 1024 if torch.cuda.is_available() else 512  # mac support
unique_token = "sks"
log_dir = f"runs/sdxl-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
device = "cuda" if torch.cuda.is_available() else "mps"  # mac support
# === Load components ===
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
tokenizer_1 = pipe.tokenizer
tokenizer_2 = pipe.tokenizer_2
text_encoder_1 = pipe.text_encoder
text_encoder_2 = pipe.text_encoder_2
vae = pipe.vae
unet = pipe.unet
noise_scheduler = pipe.scheduler

# === LoRA ===
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    #task_type=TaskType.UNSPECIFIED,
)
unet = get_peft_model(unet, lora_config)

# === Dataset ===
class SDXLDataset(Dataset):
    def __init__(self, image_dir, metadata_file):
        with open(metadata_file, 'r') as f:
            self.meta = [json.loads(line) for line in f]
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        image = Image.open(os.path.join(self.image_dir, item['file_name'])).convert("RGB")
        text = item['text']
        return {
            "pixel_values": self.transform(image),
            "text": text
        }

dataset = SDXLDataset(os.path.join(dataset_path, "images"), os.path.join(dataset_path, "metadata.jsonl"))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
accelerator = Accelerator()
unet, dataloader = accelerator.prepare(unet, dataloader)
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

# === Training loop ===
for epoch in range(1):
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
        text_input = batch["text"]

        # Latents
        latents = vae.encode(pixel_values).latent_dist.sample() * 0.13025

        # Noise & timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Text encoding — only text_encoder_2 is needed
        # Use SDXL pipeline's internal logic to encode the prompt properly
        prompt = batch["text"]
        prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(prompt)
        
        # Time IDs: SDXL requires original, crop, and target sizes
        add_time_ids = torch.tensor([[
            resolution, resolution,  # original_size
            0, 0,                    # crop coords
            resolution, resolution   # target_size
        ]], dtype=torch.int32, device=prompt_embeds.device)
        add_time_ids = add_time_ids.expand(prompt_embeds.shape[0], -1)
        
        # Predict noise
        model_pred = unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        ).sample
        
        loss = torch.nn.functional.mse_loss(model_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar("loss", loss.item(), epoch * len(dataloader) + step)

        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

writer.close()
unet.save_pretrained("lora-sdxl-trained")

