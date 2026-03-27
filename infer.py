import os
from itertools import islice
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torch.utils.data import DataLoader
from torchvision import transforms as tvt, transforms
from torchvision import utils

from ldpc import gauss_encode, ldpc_encode, ldpc_decode, gauss_decode, latentsToWatermark, watermarkToLatents
from utils import SimpleImageDataset, load_image


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.no_grad()
def i2i_inversion(imgname: str, num_steps: int = 50, verify: Optional[bool] = False) -> torch.Tensor:
    device = 'cuda'
    dtype = torch.float16
    # "stabilityai/stable-diffusion-2-1"  "stabilityai/stable-diffusion-2"  "CompVis/stable-diffusion-v1-4"
    model_id = "stabilityai/stable-diffusion-2-1"

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae

    input_img = load_image(imgname, target_size=768).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt="man", negative_prompt="", guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)

    print(inv_latents.mean(), inv_latents.var())

    # verify
    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
        image = pipe(prompt="man", negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.show()
    return inv_latents


@torch.no_grad()
def multi_generate(redundancy=16,
                   CR=0.25,
                   snr=0,
                   cfg=7.5,
                   gen_index=-1,
                   model_id="stabilityai/stable-diffusion-2-1",
                   scheduler_id="DDIMScheduler",
                   sample_step=50,
                   inverse_step=50,
                   start_i=0,
                   sum=1000):
    # Parameters
    device = 'cuda'
    dtype = torch.float32
    # Model IDs: "stabilityai/stable-diffusion-2-1" "stabilityai/stable-diffusion-2" "CompVis/stable-diffusion-v1-4"
    prompt_max_len = 400
    batch_size = 8
    start_index = int(start_i / batch_size)
    negative_prompt = [""] * batch_size
    error_cnt = 0
    table_decision = True
    decision_cnt = 0
    z_size = 64

    # Save directories
    generated_img_path = f"eval/generated_img/{gen_index}/"
    watermark_img_path = f"eval/generated_watermark/{gen_index}/"
    if not os.path.exists(generated_img_path):
        os.makedirs(generated_img_path)
    if not os.path.exists(watermark_img_path):
        os.makedirs(watermark_img_path)

    # Dataset
    ds = load_dataset("diffusers-parti-prompts/sd-v1-5", split="train")

    def collate_fn(batch):
        prompts = [item["Prompt"] for item in batch]
        return prompts

    dataLoader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Scheduler / Sampler
    scheduler = eval(f"{scheduler_id}.from_pretrained('{model_id}', subfolder='scheduler')")
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler=scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype,
                                                   local_files_only=True).to(device)

    # Set watermark information
    message = np.zeros(256, dtype=int)

    wm, H, G = ldpc_encode(message=message, batch_size=batch_size, redundancy=redundancy, CR=CR)

    # Sampling loop
    for index, prompt in islice(enumerate(dataLoader), start_index, None):

        # Apply Watermark
        latents = torch.randn(batch_size, 4, z_size, z_size).to(device)  # Initial noise
        latents = watermarkToLatents(wm, latents)

        # Re-initialize scheduler within the loop
        pipe.scheduler = eval(f"{scheduler_id}.from_pretrained('{model_id}', subfolder='scheduler')")

        # Custom config: Ensure first-order solver consistency for robustness
        custom_config = {
            "solver_order": 1,
            "algorithm_type": "dpmsolver++",
            "lower_order_final": False,
        }
        # Update scheduler configuration
        pipe.scheduler = scheduler.from_config({**scheduler.config, **custom_config})

        # Generate images based on prompt
        img = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=cfg, num_inference_steps=sample_step,
                   latents=latents, output_type='pt', return_dict=False)[0]

        # Save generated images
        for i in range(batch_size):
            utils.save_image(
                img[i].cpu().data,
                f"{generated_img_path}{index * batch_size + i}.png",
                normalize=False,
                value_range=(0, 1),
            )

        # VAE Encoding (Image to Latents)
        vae = pipe.vae
        latents = img_to_latents(img, vae)

        # Inversion to retrieve noise
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
        inv_latents, _ = pipe(prompt=negative_prompt, negative_prompt=negative_prompt, guidance_scale=1,
                              width=img.shape[-1], height=img.shape[-2], output_type='latent', return_dict=False,
                              num_inference_steps=inverse_step, latents=latents)

        # Extract watermark from noise latents
        final_tensor = latentsToWatermark(wm.shape[1], inv_latents)
        m_batch, decision_n = ldpc_decode(final_tensor, H, G, redundancy, table_decision, snr)
        decision_cnt += decision_n

        # Log error rate for each sample
        with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
            for i, m in enumerate(m_batch):
                num_errors = np.sum(message != m)  # Count bit errors
                error_rate = num_errors / len(message)
                if error_rate > 0:
                    f.write(f"Sample {index * batch_size + i} Error Rate: {error_rate * 100:.2f}%\n")
                    error_cnt += 1

            f.write(f"Majority Voting Count: {decision_cnt}\n")

        # Finalize and break
        if (index + 1) * batch_size == sum:
            avg_error_rate = error_cnt / sum
            with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
                f.write(f"Average Error Rate: {avg_error_rate * 100:.2f}%\n")

            print("completed")
            break


@torch.no_grad()
def multi_generate_gauss(gen_index=0, scheduler_id="DDIMScheduler", FR=0.1):
    # Parameters
    device = 'cuda'
    dtype = torch.float32
    # Model IDs: "stabilityai/stable-diffusion-2-1"  "stabilityai/stable-diffusion-2"  "CompVis/stable-diffusion-v1-4"
    model_id = "stabilityai/stable-diffusion-2-1"
    prompt_max_len = 400
    batch_size = 8
    start_index = int(0 / 4)
    sum = 1000
    negative_prompt = [""] * batch_size
    num_steps = 50
    redundancy = 64
    error_cnt = 0
    z_size = 64

    # Save directories
    generated_img_path = f"eval/generated_img/{gen_index}/"
    watermark_img_path = f"eval/generated_watermark/{gen_index}/"
    if not os.path.exists(generated_img_path):
        os.makedirs(generated_img_path)
    if not os.path.exists(watermark_img_path):
        os.makedirs(watermark_img_path)

    # Dataset
    ds = load_dataset("diffusers-parti-prompts/sd-v1-5", split="train")
    # ds = load_from_disk("./dataset/diffusers-parti-prompts/sd-v2.1")
    ds = ds.filter(lambda x: len(x["Prompt"]) <= prompt_max_len)

    def collate_fn(batch):
        prompts = [item["Prompt"] for item in batch]
        return prompts

    dataLoader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Scheduler / Sampler
    # Options: DDIMScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler
    scheduler = eval(f"{scheduler_id}.from_pretrained('{model_id}', subfolder='scheduler')")

    # Model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None,
                                                   torch_dtype=dtype).to(device)

    # Obtain watermark info and set redundancy
    # message = np.random.randint(2, size=256)  # 256-bit, 0 or 1
    message = np.zeros(256, dtype=int)

    wm = gauss_encode(message=message, batch_size=batch_size, redundancy=redundancy)

    # Sampling loop
    for index, prompt in islice(enumerate(dataLoader), start_index, None):
        # Apply Watermark
        latents = torch.randn(batch_size, 4, z_size, z_size).to(device)  # Initial noise
        latents = watermarkToLatents(wm, latents)

        # Reset scheduler for the loop
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')

        # Generate images based on prompt
        img = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=7.5, num_inference_steps=num_steps,
                   latents=latents, output_type='pt', return_dict=False)[0]

        # Save generated images
        for i in range(batch_size):
            utils.save_image(
                img[i].cpu().data,
                f"{generated_img_path}{index * batch_size + i}.png",
                normalize=False,
                value_range=(0, 1),
            )

        # VAE Encoding (Image to Latents)
        vae = pipe.vae
        latents = img_to_latents(img, vae)

        # Inversion to retrieve noise
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
        inv_latents, _ = pipe(prompt=negative_prompt, negative_prompt=negative_prompt, guidance_scale=1,
                              width=img.shape[-1], height=img.shape[-2], output_type='latent', return_dict=False,
                              num_inference_steps=num_steps, latents=latents)

        # Extract watermark from noise latents
        final_tensor = latentsToWatermark(wm.shape[1], inv_latents)

        m_batch = gauss_decode(final_tensor, redundancy)

        # Log error rate for each sample
        with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
            # Calculate error rate for each decoded sample
            for i, m in enumerate(m_batch):
                num_errors = np.sum(message != m)  # Count bit errors
                error_rate = num_errors / len(message)  # Calculate error rate
                if error_rate > 0:
                    f.write(f"Sample {index * batch_size + i} Error Rate: {error_rate * 100:.2f}%\n")
                    error_cnt += 1

        # Finalize and break
        if (index + 1) * batch_size == sum:
            avg_error_rate = error_cnt / sum
            with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
                f.write(f"Average Error Rate: {avg_error_rate * 100:.2f}%\n")

            print("completed")
            break


@torch.no_grad()
def robustness_ldpc_test(img_path,
                         gen_index,
                         redundancy=16, CR=0.25, snr=0,
                         model_id="stabilityai/stable-diffusion-2-1",
                         inverse_step=50,
                         start_i=0,
                         sum=96):
    # Parameters
    device = 'cuda'
    dtype = torch.float32
    batch_size = 8
    start_index = int(start_i / batch_size)
    negative_prompt = [""] * batch_size
    error_cnt = 0
    table_decision = True
    decision_cnt = 0

    # Save directories
    # visual_img_path = f"eval/visual_img/{gen_index}/"
    # if not os.path.exists(visual_img_path):
    #     os.makedirs(visual_img_path)
    watermark_img_path = f"eval/generated_watermark/{gen_index}/"
    if not os.path.exists(watermark_img_path):
        os.makedirs(watermark_img_path)

    # Dataset setup
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
    ])
    dataset = SimpleImageDataset(img_path, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Scheduler / Sampler
    scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    # Model initialization
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None,
                                                   torch_dtype=dtype).to(device)

    # Get watermark information
    # message = np.random.randint(2, size=256)  # 256-bit, 0 or 1
    message = np.zeros(256, dtype=int)

    wm, H, G = ldpc_encode(message=message, batch_size=batch_size, redundancy=redundancy, CR=CR)

    # Sampling loop
    for index, img in islice(enumerate(dataLoader), start_index, None):
        img = img.to(device)

        # VAE Encoding (Image to Latents)
        vae = pipe.vae
        latents = img_to_latents(img, vae)

        # Inversion to retrieve noise
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
        inv_latents, _ = pipe(prompt=negative_prompt, negative_prompt=negative_prompt, guidance_scale=1,
                              width=img.shape[-1], height=img.shape[-2], output_type='latent', return_dict=False,
                              num_inference_steps=inverse_step, latents=latents)

        # Extract watermark from noise latents
        final_tensor = latentsToWatermark(wm.shape[1], inv_latents)

        # ldpc_decode_t(final_tensor, H, G, redundancy, table_decision, snr, visual_img_path, batch_size, index)
        # continue

        m_batch, decision_n = ldpc_decode(final_tensor, H, G, redundancy, table_decision, snr)
        decision_cnt += decision_n

        # Log error rates for each sample
        with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
            # Calculate error rate for each decoded sample
            for i, m in enumerate(m_batch):
                num_errors = np.sum(message != m)  # Count bit errors
                error_rate = num_errors / len(message)  # Calculate error rate
                if error_rate > 0:
                    f.write(f"Sample {index * batch_size + i} Error Rate: {error_rate * 100:.2f}%\n")
                    error_cnt += 1

            f.write(f"Majority Voting Count: {decision_cnt}\n")

        # Finalize results
        if (index + 1) * batch_size == sum:
            avg_error_rate = error_cnt / sum
            with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
                f.write(f"Average Error Rate: {avg_error_rate * 100:.2f}%\n")

            print("completed")
            break


@torch.no_grad()
def robustness_gauss_test(img_path, gen_index, start_i=0, sum=1000, model_id="stabilityai/stable-diffusion-2-1"):
    # Parameters
    device = 'cuda'
    dtype = torch.float32
    # Model IDs: "stabilityai/stable-diffusion-2-1" "stabilityai/stable-diffusion-2" "CompVis/stable-diffusion-v1-4"

    batch_size = 8
    start_index = int(start_i / batch_size)
    negative_prompt = [""] * batch_size
    num_steps = 50
    gen_index = gen_index
    redundancy = 64
    error_cnt = 0

    # Save directories
    generated_img_path = f"eval/generated_img/{gen_index}/"
    watermark_img_path = f"eval/generated_watermark/{gen_index}/"
    if not os.path.exists(generated_img_path):
        os.makedirs(generated_img_path)
    if not os.path.exists(watermark_img_path):
        os.makedirs(watermark_img_path)

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
    ])
    dataset = SimpleImageDataset(img_path, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Sampler / Scheduler
    scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')

    # Model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None,
                                                   torch_dtype=dtype).to(device)

    # Obtain watermark information and set redundancy
    # message = np.random.randint(2, size=256)  # 256-bit, 0 or 1
    message = np.zeros(256, dtype=int)

    wm = gauss_encode(message=message, batch_size=batch_size, redundancy=redundancy)

    # Sampling loop
    for index, img in islice(enumerate(dataLoader), start_index, None):
        img = img.to(device)

        # VAE Encoding (Image to Latents)
        vae = pipe.vae
        latents = img_to_latents(img, vae)

        # Inversion to retrieve noise
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
        inv_latents, _ = pipe(prompt=negative_prompt, negative_prompt=negative_prompt, guidance_scale=1,
                              width=img.shape[-1], height=img.shape[-2], output_type='latent', return_dict=False,
                              num_inference_steps=num_steps, latents=latents)

        # Extract watermark from noise latents
        final_tensor = latentsToWatermark(wm.shape[1], inv_latents)

        m_batch = gauss_decode(final_tensor, redundancy)

        # Log error rates for each sample
        with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
            # Calculate error rate for each decoded sample
            for i, m in enumerate(m_batch):
                num_errors = np.sum(message != m)  # Count bit errors
                error_rate = num_errors / len(message)  # Calculate error rate
                if error_rate > 0:
                    f.write(f"Sample {index * batch_size + i} Error Rate: {error_rate * 100:.2f}%\n")
                    error_cnt += 1

        # Finalize and break
        if (index + 1) * batch_size == sum:
            avg_error_rate = error_cnt / sum
            with open(watermark_img_path + 'decoding_errors.txt', 'a') as f:
                f.write(f"Average Error Rate: {avg_error_rate * 100:.2f}%\n")

            print("completed")
            break
