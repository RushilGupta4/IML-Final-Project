import os
import torch
import math
import numpy as np
from PIL import Image, ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    DDIMScheduler,
)

# -----------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------
CHECKPOINT_DIR = "output/epoch-241"
CHECKPOINT_DIR = "output_large/epoch-451"
# CHECKPOINT_DIR = "output_large/epoch-351"
# CHECKPOINT_DIR = "output_2/epoch-221"

PROMPT = "a perfect red rose with clear petals"
# PROMPT = "a dog playing with a ball"
# PROMPT = "a brown dog"
PROMPT = "a beautiful snowy mountain landscape"
PROMPT = "a cow"
# PROMPT = "an empty table"
# PROMPT = "a snowy beach"
PROMPT = "A tennis court"
PROMPT = "a red rose"
# PROMPT = "A bicycle in the park"
# PROMPT = ""
PROMPT = "a beach"

NUM_STEPS = 50
GUIDANCE_SCALE = 7.5
# GUIDANCE_SCALE = 5
SEED = 42
# SEED = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 128
NUM_IMAGES = 4
OUTPUT_PATH = "sample.png"
GRID_PADDING = 5  # White padding between images in pixels
BORDER_WIDTH = 1  # Black border width in pixels
# -----------------------------------------------------------------

BASE_MODEL = "CompVis/stable-diffusion-v1-4"  # pretrained VAE & CLIP source


def build_pipeline(
    checkpoint_dir: str, device: torch.device
) -> StableDiffusionPipeline:
    """Re-assemble Stable Diffusion with your fine-tuned UNet."""
    # 1. Frozen components
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder"
    ).to(device)

    # 2. Fine-tuned UNet
    unet_path = os.path.join(checkpoint_dir, "unet_ema")
    unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)

    print("Number of UNet parameters:", unet.num_parameters())

    # 3. Scheduler (must match training settings)
    scheduler = DDIMScheduler(
        num_train_timesteps=1_000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        clip_sample=False,
        steps_offset=1,
        timestep_spacing="leading",
    )

    # 4. Assemble pipeline
    pipe = StableDiffusionPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=False)  # show tqdm bar
    pipe.enable_attention_slicing()  # memory-friendly
    return pipe


def create_image_grid(images, padding=10, border_width=2):
    """Create a grid of images with padding and borders."""
    # Determine grid dimensions
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))

    # Get individual image dimensions
    img_width, img_height = images[0].size

    # Calculate grid dimensions with padding
    grid_width = grid_size * (img_width + 2 * border_width) + (grid_size - 1) * padding
    grid_height = (
        grid_size * (img_height + 2 * border_width) + (grid_size - 1) * padding
    )

    # Create white background
    grid_img = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid_img)

    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size

        # Calculate position with padding
        x = col * (img_width + 2 * border_width + padding)
        y = row * (img_height + 2 * border_width + padding)

        # Draw black border
        draw.rectangle(
            [
                x,
                y,
                x + img_width + 2 * border_width - 1,
                y + img_height + 2 * border_width - 1,
            ],
            fill="black",
        )

        # Place image inside border
        grid_img.paste(img, (x + border_width, y + border_width))

    return grid_img


def main():
    if SEED is not None:
        torch.manual_seed(SEED)

    device = torch.device(DEVICE)
    pipe = build_pipeline(CHECKPOINT_DIR, device)

    # Generate all images at once using num_images_per_prompt
    with torch.no_grad():
        result = pipe(
            PROMPT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_images_per_prompt=NUM_IMAGES,
        )
        all_images = result.images

    grid_img = create_image_grid(
        all_images, padding=GRID_PADDING, border_width=BORDER_WIDTH
    )
    grid_img.save(OUTPUT_PATH)
    print(
        f"✓ Grid image saved to {OUTPUT_PATH} | {grid_img.size[0]}x{grid_img.size[1]} pixels"
    )


if __name__ == "__main__":
    main()
