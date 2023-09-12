#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import itertools
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami

from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.cross_attention import CrossAttention
from utils.custom_utils import CustomDiffusionAttnProcessor, set_use_memory_efficient_attention_xformers
from pipeline_stable_diffusion import StableDiffusionPipeline

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

logger = get_logger(__name__)


def log_validation(args, pipeline, accelerator, epoch, validation_prompt):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {validation_prompt}."
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            images += pipeline(validation_prompt, num_inference_steps=25, generator=generator).images
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    del pipeline
    torch.cuda.empty_cache()



def create_custom_diffusion(unet, freeze_model):
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif freeze_model == "crossattn_kv":
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        else:
            raise ValueError(
                    "freeze_model argument only supports crossattn_kv or crossattn"
                )

    # change attn class
    def change_attn(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = set_use_memory_efficient_attention_xformers.__get__(layer, layer.__class__)
                setattr(layer, 'set_use_memory_efficient_attention_xformers', bound_method)
            else:
                change_attn(layer)

    change_attn(unet)
    unet.set_attn_processor(CustomDiffusionAttnProcessor())
    return unet


class SingleImageDataset(Dataset):
    def __init__(
        self,
        prompt,
        mask,
        tokenizer,
        repeats,
        reg_dirs,
        with_prior_preservation,
        modifier_tokens,
        text_inv
    ):
        self.tokenizer = tokenizer
        self.text_inv = text_inv
        self.prompt = [prompt, f"a photo of a {modifier_tokens[0]}"]
        self._length = 1*repeats # number of images * repeats
        self.mask = torch.Tensor(mask).bool().repeat(4,1,1)
        self.with_prior_preservation = with_prior_preservation
        self.class_images_path = []

        self.input_ids = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if with_prior_preservation:
            for reg_dir in reg_dirs:
                with open(f"{reg_dir}/images.txt", "r") as f:
                    class_images_path = f.read().splitlines()
                with open(f"{reg_dir}/caption.txt", "r") as f:
                    class_prompt = f.read().splitlines()
                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path)
            random.shuffle(self.class_images_path)
            self.num_class_images = len(self.class_images_path)
            print("num of class image: ",self.num_class_images)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        example = {}
        example["mask"] = self.mask
        if self.text_inv:
            example["input_ids"] = random.choice(self.input_ids)
        else:
            example["input_ids"] = self.input_ids[0]

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[idx % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def add_tokens_for_inversion(tokenizer, text_encoder, modifier_tokens, initializer_tokens):
    # Add the placeholder token in tokenizer
    modifier_token_ids = []
    initializer_token_ids = []
    for modifier_token,initializer_token in zip(modifier_tokens,initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(modifier_token)
        if num_added_tokens == 0 and modifier_token != initializer_token:
            raise ValueError(
                f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                " `modifier_token` that is not already in the tokenizer."
            )

        # Convert the initializer_token, modifier_token to ids
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)

        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        modifier_token_id = tokenizer.convert_tokens_to_ids(modifier_token)
        initializer_token_ids.append(initializer_token_id)
        modifier_token_ids.append(modifier_token_id)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for modifier_token_id, initializer_token_id in zip(modifier_token_ids, initializer_token_ids):
        token_embeds[modifier_token_id] = token_embeds[initializer_token_id]
    return modifier_token_ids

def get_processed_image(image_path, size=512, interpolation="bicubic"):
    interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
    }[interpolation]
    
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")

    image = image.resize((size, size), resample=interpolation)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image


def run(args):

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # Load vae and unet
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze vae and move to device
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("Single Image Learning")
    
    # Convert images to latent space
    pixel_values = get_processed_image(args.image_path, size = args.resolution).to(accelerator.device)
    latents = vae.encode(pixel_values.to(dtype = weight_dtype)).latent_dist.sample().detach()
    latents = latents * vae.config.scaling_factor

    # Run textual inversion before fine-tuning
    stages = [f"Textual Inversion {modifier_token}" for modifier_token in args.iv_modifier_tokens]
    stages.append("Fine Tuning")
    to_merge = {}
    for stage_i, stage in enumerate(stages):
        logger.info(f"***** Running {stage} *****")
        if stage != "Fine Tuning":
            text_inv = True
            learning_rate, train_batch_size, max_train_steps, gradient_accumulation_steps, with_prior_preservation = \
                    args.iv_lr, args.iv_train_batch_size, args.iv_max_train_steps, args.iv_gradient_accumulation_steps, False
            initializer_tokens = [args.iv_initializer_tokens[stage_i]]
            embed_output_dir = os.path.join(args.output_dir, initializer_tokens[0])
            modifier_tokens = [args.iv_modifier_tokens[stage_i]]
            validation_prompt =  f"a photo of a {modifier_tokens[0]}"
            mask = args.iv_mask[stage_i]
            unet.requires_grad_(False)
        else:
            text_inv = False
            learning_rate, train_batch_size, max_train_steps, gradient_accumulation_steps, with_prior_preservation = \
                    args.ft_lr, args.ft_train_batch_size, args.ft_max_train_steps, args.ft_gradient_accumulation_steps, args.with_prior_preservation
            initializer_tokens = args.ft_initializer_tokens
            embed_output_dir = os.path.join(args.output_dir, "fine_tune")
            modifier_tokens = args.ft_modifier_tokens
            validation_prompt = args.train_prompt[-1]
            mask = np.ones((1,64,64),dtype=bool)
            unet = create_custom_diffusion(unet, args.freeze_model)

        accelerator.gradient_accumulation_steps = gradient_accumulation_steps
        os.makedirs(embed_output_dir, exist_ok = True)

        # recalculate number of training epochs
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
        # Log training informations
        if not text_inv and with_prior_preservation:
            # The batch size is doubled because prior images are added
            num_train_epochs = math.ceil(max_train_steps / (gradient_accumulation_steps * train_batch_size * 2))
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {train_batch_size*2}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size*2}")
        else:
            num_train_epochs = math.ceil(max_train_steps / (gradient_accumulation_steps * train_batch_size))
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")

        # Set up parameters to optimize
        if modifier_tokens:
            modifier_token_ids = add_tokens_for_inversion(tokenizer, text_encoder, modifier_tokens,initializer_tokens)
            if text_inv:
                params_to_optimize = text_encoder.get_input_embeddings().parameters()
            else:
                params_to_optimize = itertools.chain(text_encoder.get_input_embeddings().parameters() , [x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])] )
        else:
            modifier_token_ids = None
            params_to_optimize = itertools.chain([x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])] )
            
        if args.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            unet.train()
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        if args.scale_lr:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            params_to_optimize, 
            lr = learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = SingleImageDataset(
            prompt = args.train_prompt[stage_i],
            mask = mask,
            tokenizer = tokenizer,
            repeats = num_train_epochs*train_batch_size*gradient_accumulation_steps,
            reg_dirs = args.reg_dirs,
            with_prior_preservation = with_prior_preservation,
            modifier_tokens = modifier_tokens,
            text_inv = text_inv,
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
        )

        if latents.shape[0] < train_batch_size:
            # Repeat the latents to match batch_size
            latents = latents.repeat_interleave(train_batch_size, dim=0)
        else:
            latents = latents[:train_batch_size]

        # Scheduler
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )

        # Prepare everything with our `accelerator`.
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # Move unet to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)

        # keep original embeddings as reference
        orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
        
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(num_train_epochs), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        progress_bar.reset()

        global_step = 0
        # Run training!
        for batch in train_dataloader:
            text_encoder.train()
            with accelerator.accumulate(text_encoder):
                if with_prior_preservation:
                    reg_latents = vae.encode(batch["class_images"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    latents = torch.cat([latents,reg_latents])
                    input_ids = torch.cat([batch["input_ids"], batch["class_prompt_ids"]])
                else:
                    input_ids = batch["input_ids"]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                mask = batch["mask"]
                if text_inv:
                    loss = F.mse_loss(model_pred[mask].float(), target[mask].float(), reduction="mean")
                else:
                    # fine tuning
                    if with_prior_preservation:
                        latents, _ = torch.chunk(latents, 2, dim=0)
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)
                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if modifier_tokens is not None:
                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates = torch.arange(len(tokenizer)) != modifier_token_ids[0]
                    for i in range(len(modifier_token_ids[1:])):
                        index_no_updates = index_no_updates & (torch.arange(len(tokenizer)) != modifier_token_ids[i])
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)


            if global_step >= max_train_steps:
                break

        if text_inv:
            learned_embeds = text_encoder.get_input_embeddings().weight[modifier_token_ids[0]]
            to_merge[modifier_tokens[0]] = learned_embeds

        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_path = os.path.join(embed_output_dir, "delta.bin")
            pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            tokenizer=tokenizer,
                            revision=args.revision)
            
            pipeline.save_pretrained(save_path, only_text_inv=text_inv, to_merge=to_merge, modifier_tokens = modifier_tokens)
            log_validation(args, pipeline, accelerator, num_train_epochs, validation_prompt)

            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()



