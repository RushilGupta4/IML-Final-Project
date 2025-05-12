import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, EMAModel


# Global Configuration Parameters
SEED = 42
DATASET_ROOT = "coco"
OUTPUT_DIR = "output"
IMAGE_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 750
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "no"
NUM_WORKERS = 4
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
CAPTION_DROPOUT_PROB = 0.20
EMA_DECAY = 0.9999
LOAD_CHECKPOINT = "output_large/epoch-151"
LOAD_CHECKPOINT = None
SAVE_EVERY = 10

# Model Hyperparameters
PRETRAINED_MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
SMALLER_UNET_CONFIG = {
    "sample_size": IMAGE_SIZE // 8,
    "in_channels": 4,
    "out_channels": 4,
    "down_block_types": (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    "block_out_channels": (320, 640, 960, 960),  # Large
    "layers_per_block": 2,
    "cross_attention_dim": 768,
    "attention_head_dim": 8,
    "use_linear_projection": True,
    "only_cross_attention": False,
    "resnet_time_scale_shift": "scale_shift",
}

# Setup for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False


def coco_collate_fn(batch):
    """
    Selects one caption per image (a random one) and stacks the image tensors.
    """
    images = torch.stack([item[0] for item in batch])
    captions = [random.choice(item[1]) for item in batch]
    return images, captions


def get_dataloader(batch_size=128, num_workers=4, root="coco", train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Define annotation file and image folder depending on the train flag.
    if train:
        ann_file = f"{root}/annotations/captions_train2014.json"
        img_folder = f"{root}/train2014"
    else:
        ann_file = f"{root}/annotations/captions_val2014.json"
        img_folder = f"{root}/val2014"

    # Create the COCO Captions dataset.
    dataset = datasets.CocoCaptions(
        root=img_folder, annFile=ann_file, transform=transform
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Loading images from: {img_folder}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=coco_collate_fn,
        persistent_workers=True,
        pin_memory=True,
    )
    return dataloader


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
        kwargs_handlers=[ddp_kwargs],
    )

    # Load pretrained VAE
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME, subfolder="vae")
    vae.requires_grad_(False)  # Freeze VAE parameters
    vae.eval()

    # Load pretrained CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME, subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)  # Freeze text encoder parameters
    text_encoder.eval()

    # Create a smaller UNet model
    unet = UNet2DConditionModel(**SMALLER_UNET_CONFIG)

    # Initialize UNet weights (instead of loading pretrained ones)
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    unet.apply(init_weights)

    ema_unet = EMAModel(parameters=unet.parameters(), decay=EMA_DECAY)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1_000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",  # keep explicit for clarity
        clip_sample=False,
        steps_offset=1,  # <- NEW: future-proof
        timestep_spacing="leading",  # matches SD conventions
    )

    interefence_scheduler = DDIMScheduler(
        num_train_timesteps=1_000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",  # keep explicit for clarity
        clip_sample=False,
        steps_offset=1,  # <- NEW: future-proof
        timestep_spacing="leading",  # matches SD conventions
    )

    # Create optimizer
    optimizer = optim.AdamW(
        unet.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Get dataloader
    train_dataloader = get_dataloader(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, root=DATASET_ROOT, train=True
    )

    # Prepare models for accelerator
    unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    )
    ema_unet.to(accelerator.device)

    # Training loop
    start_epoch = 0
    global_step = 0

    if LOAD_CHECKPOINT is not None and os.path.isdir(LOAD_CHECKPOINT):
        accelerator.print(f"Resuming from {LOAD_CHECKPOINT}")
        accelerator.load_state(LOAD_CHECKPOINT)  # restores model/opt/scheduler

        ema_ckpt = os.path.join(LOAD_CHECKPOINT, "unet_ema", "pytorch_model.bin")
        if os.path.exists(ema_ckpt):
            ema_unet.load_state_dict(torch.load(ema_ckpt, map_location="cpu"))

        state_file = os.path.join(LOAD_CHECKPOINT, "trainer_state.pt")
        if os.path.exists(state_file):
            state = torch.load(state_file, map_location="cpu")
            start_epoch = state.get("epoch", 0) + 1
            global_step = state.get("global_step", 0)
            accelerator.print(
                f" → Starting at epoch {start_epoch}, global step {global_step}"
            )

    for epoch in range(start_epoch, EPOCHS):
        unet.train()
        train_loss = 0.0
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            disable=not accelerator.is_main_process,
        )

        for step, (images, captions) in enumerate(train_dataloader):
            drop_mask = [random.random() < CAPTION_DROPOUT_PROB for _ in captions]
            captions_for_tokeniser = [
                "" if drop else cap for cap, drop in zip(captions, drop_mask)
            ]

            with accelerator.accumulate(unet):
                # Encode text using CLIP
                text_inputs = tokenizer(
                    captions_for_tokeniser,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)

                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = text_encoder(text_inputs.input_ids)[0]

                # Encode image into latent space using VAE
                with torch.no_grad():
                    with accelerator.autocast():
                        latents = vae.encode(
                            images.to(accelerator.device)
                        ).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor  # Scale latents

                # Sample noise
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the model prediction for the noise
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

                # Calculate loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    loss = F.mse_loss(noise_pred, noise)
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    loss = F.mse_loss(noise_pred, target)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Update weights
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()
                    ema_unet.step(unet.parameters())
                    optimizer.zero_grad()

            # Update progress bar and log metrics
            train_loss += loss.detach().item()
            global_step += 1
            progress_bar.update(1)

            # Log training metrics
            avg_loss = train_loss / (step + 1)
            lr = optimizer.param_groups[0]["lr"]
            logs = {
                "loss": loss.detach().item(),
                "train_loss": avg_loss,
                "lr": lr,
            }

            progress_bar.set_postfix(**logs)

        lr_scheduler.step()

        # End of epoch
        avg_epoch_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            ema_unet.store(unet.parameters())  # save current
            ema_unet.copy_to(unet.parameters())  # load EMA → unet
            accel_unet = accelerator.unwrap_model(unet).to(accelerator.device)

            if epoch % SAVE_EVERY == 0:
                # Save EMA model
                accel_unet.save_pretrained(os.path.join(checkpoint_dir, "unet_ema"))

                # accelerator state (model/opt/etc.)
                accelerator.save_state(checkpoint_dir)
                torch.save(
                    {"epoch": epoch, "global_step": global_step},
                    os.path.join(checkpoint_dir, "trainer_state.pt"),
                )

            generate_samples(
                accel_unet,
                vae,
                text_encoder,
                tokenizer,
                interefence_scheduler,
                checkpoint_dir,
            )
            ema_unet.restore(unet.parameters())

    # End of training
    if accelerator.is_main_process:
        # Save final model
        final_dir = os.path.join(OUTPUT_DIR, "final")
        os.makedirs(final_dir, exist_ok=True)

        ema_unet.store(unet.parameters())  # save current
        ema_unet.copy_to(unet.parameters())  # load EMA → unet
        accel_unet = accelerator.unwrap_model(unet).to(accelerator.device)
        accel_unet.save_pretrained(os.path.join(final_dir, "unet_ema"))

        accelerator.save_state(final_dir)  # accelerator state (model/opt/etc.)
        torch.save(
            {"epoch": epoch, "global_step": global_step},
            os.path.join(final_dir, "trainer_state.pt"),
        )

        generate_samples(
            accel_unet,
            vae,
            text_encoder,
            tokenizer,
            interefence_scheduler,
            final_dir,
        )
        ema_unet.restore(unet.parameters())

        print("Training complete!")


def generate_samples(unet, vae, text_encoder, tokenizer, noise_scheduler, output_dir):
    """Generate sample images using the current model."""
    unet.eval()
    vae.eval()
    text_encoder.eval()

    # Sample prompts for visualization
    prompts = [
        "A photo of a rose",
        "A cat with a hat",
        "A bicycle in the park",
        "A dog playing with a ball",
    ]

    # Create pipeline
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,  # No safety checker for simplicity
        feature_extractor=None,
        requires_safety_checker=False,
    )

    # Set to inference mode
    pipeline.to(unet.device)
    pipeline.set_progress_bar_config(disable=True)

    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    # Generate images for each prompt
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            image = pipeline(
                prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]

            # Save image
            image_path = os.path.join(output_dir, "samples", f"sample_{i}.png")
            image.save(image_path)

    # Set models back to training mode
    unet.train()


if __name__ == "__main__":
    main()
