import torch
import os
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

class ImageProcessor:
    @staticmethod
    def pil_to_numpy(images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        # Ensure images are in RGB mode
        images = [image.convert('RGB') for image in images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)  # Shape: (N, H, W, C)
        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        if images.ndim == 3:
            images = images[None, ...]  # Add batch dimension
        images = images.transpose(0, 3, 1, 2)  # Shape: (N, C, H, W)
        images = torch.from_numpy(images)
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = images.cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # Shape: (N, H, W, C)
        images = np.clip(images, 0, 1)  # Ensure values are in [0, 1]
        # Remove multiplication by 255 here
        return images  # Values remain in [0, 1]

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> Union[List[Image.Image], Image.Image]:
        if images.ndim == 3:
            images = images[None, ...]  # Add batch dimension
        # Multiply by 255 only once here
        images = (images * 255).astype(np.uint8)
        # Ensure images have the correct color mode
        pil_images = [Image.fromarray(image, mode='RGB') for image in images]
        if len(pil_images) == 1:
            return pil_images[0]
        else:
            return pil_images

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        print("1")
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        print("2")
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        print("3")
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def return_img(local_latent, pipe):
    local_latent = local_latent.clone().detach()
    local_latent.requires_grad_(False)
    image_local = pipe.decode_latents(local_latent)
    return ImageProcessor.numpy_to_pil(image_local)

def return_img(local_latent, pipe):
    local_latent = local_latent.clone().detach()
    local_latent.requires_grad_(False)
    image_local = pipe.vae.decode(local_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image_local = image_local.detach()
    return pipe.image_processor.postprocess(image_local)[0]

def one_step_sds_orange(image, depth, total_epochs, pipe, view_cut):
    depth = depth.to(pipe.device)
    cur_t = [0.02, 0.98]
    clip = 1
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor = image_tensor.to(pipe.device)
    init_latents = pipe.vae.encode(image_tensor)
    init_latents = retrieve_latents(init_latents)
    init_latents = pipe.vae.config.scaling_factor * init_latents
    init_latents = init_latents.detach().clone().requires_grad_(True)
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.1)  # Choose a learning rate
    for e in range(total_epochs):
        optimizer.zero_grad()
        step_ratio = min(1, e / total_epochs)
        grad = pipe.get_sds_latent(
            f"a photo of the {view_cut} cross section of a pomegranate, detailed",
            image=image_tensor,
            depth_map=depth,
            strength=0.1,
            num_inference_steps=100,
            init_latents=init_latents,
            t_range = cur_t,
            guidance_scale=10,
            step_ratio=None
        )
        grad.clamp(-clip, clip)
        target = (init_latents - grad).detach()
        loss = 0.5 * F.mse_loss(init_latents.float(), target, reduction='sum') / init_latents.shape[0]
        loss.backward()
        optimizer.step()
    return return_img(init_latents, pipe)
