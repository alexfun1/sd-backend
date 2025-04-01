import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

def generate_images(model_path, prompt, output_dir, num_images=1, guidance_scale=7.5, num_inference_steps=50):
    """
    Generates images using a trained Stable Diffusion LoRA model.

    Args:
        model_path (str): Path to the trained LoRA model directory.
        prompt (str): The text prompt to generate images from.
        output_dir (str): Directory to save the generated images.
        num_images (int): Number of images to generate.
        guidance_scale (float): Guidance scale for classifier-free diffusion.
        num_inference_steps (int): Number of inference steps.
    """

    os.makedirs(output_dir, exist_ok=True)

    if 'xl' in model_path.lower():
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_path)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model_path)

    pipeline = pipeline.to("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        images = pipeline(
            prompt,
            num_images=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images

    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"generated_image_{i}.png"))
    print(f"Generated {num_images} images and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained LoRA model.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Directory to save generated images.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")

    args = parser.parse_args()

    generate_images(
        args.model_path,
        args.prompt,
        args.output_dir,
        args.num_images,
        args.guidance_scale,
        args.num_inference_steps,
    )