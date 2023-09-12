from typing import List, Optional, Union, Tuple
import torch
from utils.ptp_utils import AttentionStore, aggregate_attention
import numpy as np
from pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
#########################################
'''
    This is modified from Attend-and-Excite/pipeline_attend_and_excite.py
    https://github.com/yuval-alaluf/Attend-and-Excite
'''
#########################################

def get_out_loss(image, mask):
    mask_region = image*mask
    out_of_region_loss = 1-mask_region.sum()/image.sum()
    return out_of_region_loss

class LayoutPipeline(StableDiffusionPipeline):
    @staticmethod
    def _compute_out_losses(masks: List,
                            attention_maps: torch.Tensor,
                            indices_to_alter: List[int]) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        attention_for_text = attention_maps[:, :, 1:-1]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract out of region loss for each object
        out_losses = []
        for mask, i in zip(masks, indices_to_alter):
            image = attention_for_text[:, :, i]
            metrics = get_out_loss(image, mask)
            out_losses.append(metrics)
        return out_losses

    def _aggregate_and_get_out_losses(self, masks: List,
                                                   attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        out_losses = self._compute_out_losses(
            masks=masks,
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter)
        return out_losses

    @staticmethod
    def _compute_loss(losses: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the mean + max out of region loss. """
        loss = sum(losses)/len(losses) + max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(self,
                                           masks: List,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           max_refinement_steps: int = 20):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            out_losses = self._aggregate_and_get_out_losses(
                masks=masks,
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res)

            loss, losses = self._compute_loss(out_losses, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            print(f'\t Try {iteration}. {low_word} has the max losses of {out_losses[low_token]}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with the max loss: {out_losses[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        out_losses = self._aggregate_and_get_out_losses(
            masks=masks,
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res)
        loss, losses = self._compute_loss(out_losses, return_losses=True)
        print(f"\t Finished with max + mean loss of: {loss}")
        return loss, latents, out_losses

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def __call__(
            self,
            max_refinement_steps: int,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            masks: List = [],
            blend_dict: dict = {},
            **kwargs):
        
        text_embeddings, text_input, latents, do_classifier_free_guidance, extra_step_kwargs = self._setup_inference(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=generator,
            latents=latents, **kwargs
        )
            
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1
        
        blend_mask = blend_dict["blend_mask"].repeat(1,4,1,1)
        inversion_latents = blend_dict["inversion_latents"]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):

            with torch.enable_grad():

                latents = latents.clone().detach().requires_grad_(True)

                # Forward pass of denoising with text conditioning
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
                self.unet.zero_grad()

                # Get out region loss of each object
                out_losses = self._aggregate_and_get_out_losses(
                    masks=masks,
                    attention_store=attention_store,
                    indices_to_alter=indices_to_alter,
                    attention_res=attention_res)

                if not run_standard_sd:
                    # Calculate the mean + max loss
                    loss = self._compute_loss(out_losses)

                    # If this is an iterative refinement step, verify we have reached the desired threshold for all
                    if i in thresholds.keys() and loss > 1. - thresholds[i]:
                        del noise_pred_text
                        torch.cuda.empty_cache()
                        loss, latents, out_losses = self._perform_iterative_refinement_step(
                            masks=masks,
                            latents=latents,
                            indices_to_alter=indices_to_alter,
                            loss=loss,
                            threshold=thresholds[i],
                            text_embeddings=text_embeddings,
                            text_input=text_input,
                            attention_store=attention_store,
                            step_size=scale_factor * np.sqrt(scale_range[i]),
                            t=t,
                            attention_res=attention_res,
                            max_refinement_steps = max_refinement_steps)

                    # Perform gradient update
                    if i < max_iter_to_alter:
                        loss = self._compute_loss(out_losses)
                        if loss != 0:
                            latents = self._update_latent(latents=latents, loss=loss,
                                                          step_size=scale_factor * np.sqrt(scale_range[i]))
                        print(f'Iteration {i} | Loss: {loss:0.4f}')
                    
            # blending(for the overlap background region)
            if i < blend_dict["blend_steps"]:
                latents[blend_mask] = inversion_latents[i][blend_mask]

            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # Compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    

        outputs = self._prepare_output(latents, output_type, return_dict)
        
        return outputs
