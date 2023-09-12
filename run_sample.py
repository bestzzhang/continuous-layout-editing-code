import os
from typing import List, Dict, Tuple
import torch
import numpy as np
from PIL import Image
import argparse
import pprint
import json

from pipeline_layout import LayoutPipeline
from utils import ptp_utils, custom_utils
from utils.ptp_utils import AttentionStore
from diffusers import DDIMScheduler
from utils.ddim_inversion import create_inversion_latents

def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def run(pipe,
        prompt: str,
        guidance_scale: float,
        eta: float,
        n_inference_steps: int,
        controller: AttentionStore,
        indices_to_alter: List[int],
        generator: torch.Generator,
        run_standard_sd: bool = False,
        scale_factor: int = 20,
        thresholds: Dict[int, float] = {0:0.6, 10: 0.7, 20: 0.8}, 
        max_iter_to_alter: int = 25,
        max_refinement_steps: int = 20,
        scale_range: Tuple[float, float] = (1., 0.5),
        attention_res: int = 16,
        masks: List = [],
        blend_dict: dict = {}):
    if controller is not None:
        ptp_utils.register_attention_control(pipe, controller)
    outputs = pipe(masks=masks,
                    blend_dict=blend_dict,
                    max_refinement_steps=max_refinement_steps,
                    prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=indices_to_alter,
                    attention_res=attention_res,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    eta = eta,
                    num_inference_steps=n_inference_steps,
                    max_iter_to_alter=max_iter_to_alter,
                    run_standard_sd=run_standard_sd,
                    thresholds=thresholds,
                    scale_factor=scale_factor,
                    scale_range=scale_range)
    image = outputs.images[0]
    return image

def get_masks(fp, sem_dict_path):
    masks = []
    sem_dict = json.load(open(sem_dict_path))
    for k in list(sem_dict.keys())[:-1]:
        pil_image = Image.open(fp).resize((16,16),resample=0)
        img = np.array(pil_image)
        color = np.array(sem_dict[k])
        mask = torch.Tensor(np.all(img == color, axis=-1)).cuda()
        assert mask.sum()!=0
        masks.append(mask)
    return masks

def get_blend_mask(mask_image, target_layout, sem_dict_path):
    m1 = np.array(Image.open(mask_image))
    m2 = np.array(Image.open(target_layout))
    bg_color = np.array(json.load(open(sem_dict_path))["background"])
    blend_mask = np.all(m1 == bg_color, axis=-1) & np.all(m2 == bg_color, axis=-1)
    blend_mask = torch.from_numpy(blend_mask).cuda()
    return blend_mask

def load_model(delta_ckpt):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe = LayoutPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    
    tokenizer = pipe.tokenizer
    if delta_ckpt is not None:
        # load the custom cross-attention matrix
        custom_utils.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt)
    return pipe, tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Layout Control Inference.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for generation",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image(use for blending)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to generated images",
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        required=True,
        help="Path to mask image",
    )
    parser.add_argument(
        "--target_layout",
        type=str,
        required=True,
        help="Target Layout",
    )
    parser.add_argument(
        "--semantic_dict",
        type=str,
        required=True,
        help="semantic dict",
    )
    parser.add_argument(
        "--delta_ckpt",
        type=str,
        default=None,
        help="Path to trained model",
    )
    parser.add_argument(
        "--blend_steps",
        type=int,
        default=15,
        help="Number of blending steps",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    n_inference_steps = 50
    guidance_scale = 5
    eta = 0
    max_iter_to_alter = 25
    max_refinement_steps = 40
    scale_factor = 20
    run_standard_sd = False
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pipe, tokenizer = load_model(args.delta_ckpt)

    token_indices = get_indices_to_alter(pipe, args.prompt)
    masks = get_masks(args.target_layout, args.semantic_dict)

    blend_mask = get_blend_mask(args.mask_image, args.target_layout, args.semantic_dict)
    inversion_latents = create_inversion_latents(pipe, args.image_path, args.prompt, guidance_scale, n_inference_steps)
    blend_dict = {
        "blend_mask":blend_mask,
        "inversion_latents":inversion_latents,
        "blend_steps":args.blend_steps
    }

    for i, seed in enumerate([0,8,88,888,8888]):
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run(pipe,
                    prompt=args.prompt,
                    guidance_scale = guidance_scale,
                    n_inference_steps = n_inference_steps,
                    eta = eta,
                    controller=controller,
                    indices_to_alter= token_indices,
                    generator=g,
                    run_standard_sd=run_standard_sd,
                    scale_factor = scale_factor,
                    thresholds = {0:0.6, 10: 0.7, 20: 0.8},
                    max_iter_to_alter=max_iter_to_alter,
                    max_refinement_steps=max_refinement_steps,
                    scale_range = (1., 0.5),
                    masks = masks,
                    blend_dict = blend_dict,
                    )
                                
        image_name = os.path.join(args.output_dir, f"{seed}_prior.png")
        image.save(image_name)

