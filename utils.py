import glob
import os
import random
from itertools import islice
from pathlib import Path
from typing import Union, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler, \
    DPMSolverMultistepScheduler, EulerDiscreteScheduler, \
    StableDiffusionImg2ImgPipeline
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tvt, transforms
from tqdm import tqdm

from infer import img_to_latents
from ldpc import ldpc_encode, latentsToWatermark
from ldpc import pseudo_random_recover_sign_np
from utils import SimpleImageDataset, visual_error_bits


class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def calculate_accuracy(folder_paths, cnt=1000):
    """
    Calculate average accuracy across multiple folders
    1. For each folder: accumulate error rate/1000 to get the folder's error rate
    2. Average the error rates across all folders
    3. Return 1 - average error rate
    4. Count files with error rate exceeding 35%

    Args:
        folder_paths (list): List containing multiple folder paths

    Returns:
        float: Average accuracy (1 - average error rate)
    """
    folder_error_rates = []  # Store error rate for each folder
    valid_folder_count = 0  # Valid folder count
    high_error_file_count = 0  # Count of files with error rate exceeding 35%

    for folder_path in folder_paths:
        file_path = os.path.join(folder_path, 'decoding_errors.txt')

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist, skipping")
            continue

        valid_folder_count += 1
        total_error_rate = 0.0
        error_count = 0

        try:
            with open(file_path, 'r', encoding='gbk') as file:
                for line in file:
                    line = line.strip()
                    if "sample" in line and "error_rate:" in line:
                        try:
                            error_rate_str = line.split("error_rate:")[1].split("%")[0]
                            error_rate = float(error_rate_str)
                            total_error_rate += error_rate
                            error_count += 1

                            # Check if this file's error rate exceeds 35%
                            if error_rate > 35:
                                high_error_file_count += 1
                                print(f"Error rate: {error_rate:.2f}%, {error_rate:.2f} > 35, count increased by one")

                        except (IndexError, ValueError) as e:
                            print(f"Warning: Unable to parse line '{line}', skipping. Error: {e}")

            # Calculate this folder's error rate (accumulated error rate / cnt)
            folder_error_rate = total_error_rate / cnt / 100
            folder_error_rates.append(folder_error_rate)

            print(f"Folder {folder_path} statistics: Cumulative error rate={total_error_rate}% "
                  f"({error_count} samples total), Folder error rate={folder_error_rate * 100:.2f}%")

        except Exception as e:
            print(f"Warning: Error reading file {file_path}: {e}")
            continue

    if not folder_error_rates:
        print("Warning: No valid folders available for processing")
        return 0.0

    # Calculate average error rate
    average_error_rate = sum(folder_error_rates) / valid_folder_count
    accuracy = 1 - average_error_rate

    # Output statistics for files with error rate exceeding 35%
    print(f"\n=== Statistics Results ===")
    print(f"Total folders: {valid_folder_count}")
    print(f"Files with error rate > 35%: {high_error_file_count}")
    print(f"Average accuracy: {accuracy:.4f}")

    return accuracy


def visual_error_bits(message_batch, redundancy, path, batch_size, index):
    message_batch = -message_batch.cpu().numpy()  # Take negative, so 0/black pixels represent error bits
    message_batch = np.ascontiguousarray(message_batch, dtype=np.float64)  # Ensure C-contiguous

    for i, message in enumerate(message_batch):
        decoded_batch = []
        # Split each sample evenly into 16 parts
        split_messages = np.array_split(message, redundancy)

        for j, split_msg in enumerate(split_messages):
            # Pseudo-random sequence decoding
            split_code = pseudo_random_recover_sign_np(split_msg, seed=42 + j)
            split_code = np.where(split_code >= 0, 0, 1)
            decoded_batch.append(split_code)

        all_data = np.concatenate(decoded_batch)

        result = all_data.reshape(4, 64, 64)
        normalized = [((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8) for img in result]

        # Add spacing
        spacer = np.ones((64, 5), dtype=np.uint8) * 255
        combined = np.hstack([img if i == 0 else np.hstack([spacer, img]) for i, img in enumerate(normalized)])

        Image.fromarray(combined).save(path + f'{batch_size * index + i}.png')


def for_jpeg(input_folder, output_folder, qf=50):
    """
       Compress images in input folder with specified quality factor (QF) using JPEG compression,
       and save to output folder

       Args:
           input_folder (str): Input image folder path
           output_folder (str): Output folder path
           qf (int): JPEG quality factor (1-100), default 25
       """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with quality factor QF={qf}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Build output path (keep same filename but force .jpg extension)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}.jpg")

            # Save JPEG image with specified quality
            cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_rotate_square_with_black(input_folder, output_folder, angle=90):
    """
    Rotates square images and fills edges with black (maintains square output)

    Args:
        input_folder (str): Input image folder path (must contain square images)
        output_folder (str): Output folder path
        angle (int/float): Clockwise rotation angle (supports any angle)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported image formats
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_ext)]

    if not image_files:
        print(f"Warning: No image files found in {input_folder}")
        return

    print(f"Rotating {len(image_files)} square images (angle={angle}°, black fill)...")

    processed = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            # Validate square image
            h, w = img.shape[:2]
            if h != w:
                print(f"Warning: {filename} is not square ({w}x{h}), skipping")
                continue

            # Calculate rotation matrix
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Perform rotation (fill edges with black)
            rotated = cv2.warpAffine(
                img,
                M,
                (w, h),  # Keep original size
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)  # Black fill (BGR format)
            )

            # Save result (keep original format)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, rotated)
            processed += 1

        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")

    print(f"Completed! Successfully processed {processed}/{len(image_files)} images")
    print(f"Output path: {output_folder}")


def for_gaussian_blur(input_folder, output_folder, r=4):
    """
    Applies Gaussian blur to images in the input folder and saves them to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        r (int or float): Gaussian blur radius, controls blur intensity, default 4
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    # Calculate kernel_size and sigma_x based on r
    kernel_size = (int(2 * r + 1), int(2 * r + 1))  # Kernel size = 2r + 1 (must be odd)
    sigma_x = r  # sigma_x is usually equal to r

    print(
        f"Processing {len(image_files)} images with Gaussian blur parameters r={r} (kernel_size={kernel_size}, sigma_x={sigma_x})...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(img, kernel_size, sigma_x)

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, blurred_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_median_filter(input_folder, output_folder, k=7):
    """
    Applies median filter to images in the input folder and saves them to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        k (int): Median filter kernel size (must be a positive odd number), default 7
    """
    # Ensure kernel size is a positive odd number
    if k % 2 == 0:
        raise ValueError("Median filter kernel size k must be a positive odd number (e.g., 3, 5, 7)")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with median filter kernel size k={k}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Apply median filter
            filtered_img = cv2.medianBlur(img, k)

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, filtered_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_gaussian_noise(input_folder, output_folder, mu=0, sigma=0.05, alpha=1):
    """
    Adds Gaussian noise to images in the input folder and saves them to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        mu (float): Mean of Gaussian noise, default 0
        sigma (float): Standard deviation of Gaussian noise, default 0.05
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with Gaussian noise parameters µ={mu}, σ={sigma}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image (ensure loaded as float, range [0,1])
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path).astype(np.float32) / 255.0

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Generate Gaussian noise (same size as image)
            noise = np.random.normal(mu, sigma, img.shape).astype(np.float32)

            # Add noise and clip to [0,1] range
            noisy_img = img + noise * alpha
            noisy_img = np.clip(noisy_img, 0, 1)

            # Convert back to 8-bit image format [0,255]
            noisy_img = (noisy_img * 255).astype(np.uint8)

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, noisy_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_salt_pepper_noise(input_folder, output_folder, p=0.05):
    """
    Adds salt-and-pepper noise to images in the input folder and saves them to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        p (float): Total proportion of noisy pixels (0-1), default 0.05 (5% of pixels will be corrupted)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with salt-and-pepper noise total proportion p={p}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Generate salt-and-pepper noise
            noisy_img = img.copy()
            h, w, c = noisy_img.shape
            num_noise = int(p * h * w)  # Total number of noisy pixels

            # Randomly choose half for salt noise, half for pepper noise
            # The proportion of salt and pepper can also be adjusted as needed
            num_salt = num_noise // 2
            num_pepper = num_noise - num_salt  # Ensure correct total

            # Add salt noise (white pixels)
            coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
            noisy_img[coords[0], coords[1], :] = 255

            # Add pepper noise (black pixels)
            coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
            noisy_img[coords[0], coords[1], :] = 0

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, noisy_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_brightness_adjustment(input_folder, output_folder, factor=6):
    """
    Adjusts the brightness of images in the input folder and saves them to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        factor (int/float): Brightness adjustment factor (>1 for brighter, <1 for darker), default 6
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with brightness adjustment factor={factor}...")

    # Create brightness adjustment transform
    brightness_adjuster = transforms.ColorJitter(brightness=(factor, factor))

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img_bgr = cv2.imread(input_path)

            if img_bgr is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Convert BGR to RGB (as torchvision usually uses RGB)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Apply brightness adjustment
            adjusted_pil = brightness_adjuster(img_pil)

            # Convert back to numpy array and BGR format
            adjusted_rgb = np.array(adjusted_pil)
            adjusted_bgr = cv2.cvtColor(adjusted_rgb, cv2.COLOR_RGB2BGR)

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, adjusted_bgr)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_random_scale(input_folder, output_folder, scale_ratio=0.5):
    """
    Randomly scales images in the input folder (first shrinks, then enlarges) and saves them to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        scale_ratio (float): Scaling ratio (0-1), default 0.5
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with scaling ratio scale_ratio={scale_ratio}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Get original dimensions
            h, w = img.shape[:2]

            # Randomly shrink
            new_h, new_w = int(h * scale_ratio), int(w * scale_ratio)
            scaled_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Enlarge back to original dimensions
            scaled_up = cv2.resize(scaled_down, (w, h), interpolation=cv2.INTER_LINEAR)

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, scaled_up)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_random_drop(input_folder, output_folder, drop_ratio=0.2):
    """
    Randomly crops images in the input folder and fills with black, then saves to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        drop_ratio (float): Proportion to drop (0-1), default 0.2 (drops 20% area, keeps 80%)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with drop ratio drop_ratio={drop_ratio}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Get original dimensions
            h, w, _ = img.shape

            # Calculate retained dimensions (1 - drop_ratio)
            retained_h = int(h * (1 - drop_ratio))
            retained_w = int(w * (1 - drop_ratio))

            # Randomly determine crop start point
            start_h = np.random.randint(0, h - retained_h + 1)
            start_w = np.random.randint(0, w - retained_w + 1)

            # Create black background
            dropped_img = np.zeros_like(img)

            # Copy cropped region to black background
            dropped_img[start_h:start_h + retained_h, start_w:start_w + retained_w] = \
                img[start_h:start_h + retained_h, start_w:start_w + retained_w]

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, dropped_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_random_crop(input_folder, output_folder, crop_ratio=0.8):
    """
    Randomly crops images in the input folder, retaining a specified proportion, and saves them to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        crop_ratio (float): Proportion to retain (0-1), default 0.8 (retains 80% area)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with retention ratio crop_ratio={crop_ratio}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Get original dimensions
            h, w = img.shape[:2]

            # Calculate cropped dimensions
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)

            # Randomly determine crop start point
            x = np.random.randint(0, w - crop_w) if w > crop_w else 0
            y = np.random.randint(0, h - crop_h) if h > crop_h else 0

            # Directly crop image (no black fill)
            cropped_img = img[y:y + crop_h, x:x + crop_w]

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, cropped_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def for_random_crop_resize(input_folder, output_folder, crop_ratio=0.8, interpolation=cv2.INTER_LINEAR):
    """
    Randomly crops images in the input folder, then resizes them to original dimensions, and saves to the output folder

    Args:
        input_folder (str): Input image folder path
        output_folder (str): Output folder path
        crop_ratio (float): Proportion to crop (0-1), default 0.8 (retains 80% area after cropping)
        interpolation (int): Interpolation method used for resizing, default cv2.INTER_LINEAR
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"Warning: No supported image files found in input folder '{input_folder}'")
        return

    print(f"Processing {len(image_files)} images with crop ratio crop_ratio={crop_ratio}...")

    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping")
                continue

            # Get original dimensions
            h, w = img.shape[:2]

            # Calculate cropped dimensions
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)

            # Randomly determine crop start point
            x = np.random.randint(0, w - crop_w) if w > crop_w else 0
            y = np.random.randint(0, h - crop_h) if h > crop_h else 0

            # Crop image
            cropped_img = img[y:y + crop_h, x:x + crop_w]

            # Resize cropped image to original dimensions
            resized_img = cv2.resize(cropped_img, (w, h), interpolation=interpolation)

            # Build output path (keep same filename)
            output_path = os.path.join(output_folder, filename)

            # Save processed image
            cv2.imwrite(output_path, resized_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"Processing completed! Successfully processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {output_folder}")


def set_seed(seed=42):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Ensure convolutional results are deterministic
        torch.backends.cudnn.benchmark = False  # Disable benchmark acceleration

    # Other settings that may affect randomness
    os.environ['PYTHONHASHSEED'] = str(seed)


def advanced_embedding_attack_folder(
        input_folder: str,
        output_folder: str,
        sum=96
):
    """
    Simple version: Read images from a folder, perform adversarial attack, and save to another folder.
    """

    device = torch.device('cuda')

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['*.jpg', ['*.jpeg'], '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    image_paths = sorted(image_paths, key=lambda x: int(Path(x).stem))

    print(f"Found {len(image_paths)} images")

    # Image transformation
    transform = transforms.ToTensor()

    scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder='scheduler')
    encoder = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", scheduler=scheduler,
                                                      safety_checker=None,
                                                      torch_dtype=torch.float32, local_files_only=True).vae
    encoder = encoder.to(device).eval()

    for i, img_path in enumerate(image_paths):
        if i == sum:
            break
        # Read image
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Perform adversarial attack
        adversarial_image = advanced_embedding_attack(
            encoder=encoder,
            original_image=img_tensor,
            epsilon=8.0 / 255.0,
            num_iter=50,
            step_size=2.0 / 255.0,
            distance_metric='linf',
            device=device,
            verbose=False
        )

        # Save result
        adv_pil = transforms.ToPILImage()(adversarial_image.squeeze(0).cpu())
        filename = Path(img_path).name
        output_path = os.path.join(output_folder, f"{filename}")
        adv_pil.save(output_path)

        print(f"Processing complete: {filename}")

    print(f"All images processed, saved in: {output_folder}")


def advanced_embedding_attack(
        encoder,
        original_image: torch.Tensor,
        epsilon: float = 8.0 / 255.0,
        num_iter: int = 50,
        step_size: float = 2.0 / 255.0,
        distance_metric: str = 'linf',
        device: torch.device = None,
        verbose: bool = True
) -> torch.Tensor:
    """
    Simplified embedding space adversarial attack - only attacks the original image.

    Args:
        encoder: VAE encoder
        original_image: Original image
        epsilon: Perturbation upper bound
        num_iter: Number of iterations
        step_size: Step size
        distance_metric: Distance metric ('l2', 'l1', 'linf', 'cosine')
        verbose: Whether to show progress
    """

    # Set up model
    encoder = encoder.to(device).eval()
    original_image = original_image.to(device)

    # Add batch dimension if necessary
    if original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)

    # Get latent features of the original image
    with torch.no_grad():
        z_original = img_to_latents(original_image, encoder)

    # Initialize adversarial example
    x_adv = original_image.clone().detach()

    # Random initial perturbation
    random_noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = x_adv + random_noise
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    # PGD iterative attack
    for i in range(num_iter):
        x_adv.requires_grad = True

        # Forward pass to get current latent features
        z_adv = img_to_latents(original_image, encoder)

        # Calculate latent space distance
        if distance_metric == 'l2':
            distance = torch.norm(z_adv - z_original, p=2)
        elif distance_metric == 'l1':
            distance = torch.norm(z_adv - z_original, p=1)
        elif distance_metric == 'linf':
            distance = torch.max(torch.abs(z_adv - z_original))
        elif distance_metric == 'cosine':
            z_adv_flat = z_adv.flatten(1)
            z_original_flat = z_original.flatten(1)
            cosine_sim = torch.nn.functional.cosine_similarity(z_adv_flat, z_original_flat, dim=1)
            distance = 1 - cosine_sim.mean()
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Loss function: maximize latent space distance (untargeted attack)
        loss = -distance

        # Compute gradient
        grad = torch.autograd.grad(loss, [x_adv])[0]

        # PGD update
        x_adv = x_adv.detach() - step_size * grad.sign()

        # Project to constraint set
        x_adv = torch.clamp(x_adv, original_image - epsilon, original_image + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if verbose and (i + 1) % 10 == 0:
            print(f'Iteration [{i + 1}/{num_iter}], {distance_metric} distance: {distance.item():.6f}')

    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.squeeze() if original_image.shape[0] == 1 else x_adv


@torch.no_grad()
def visual_error_bits_img(img_path,
                          gen_index,
                          redundancy=16, CR=0.25, snr=0,
                          model_id="stabilityai/stable-diffusion-2-1",
                          inverse_step=50,
                          start_i=0,
                          sum=1000):
    # Parameters
    device = 'cuda'
    dtype = torch.float32
    batch_size = 8
    start_index = int(start_i / batch_size)
    negative_prompt = [""] * batch_size

    # Save directory
    visual_img_path = f"eval/visual_img/{gen_index}/"
    if not os.path.exists(visual_img_path):
        os.makedirs(visual_img_path)

    # Dataset setup
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
    ])
    dataset = SimpleImageDataset(img_path, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Scheduler setup
    scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    # Model loading
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None,
                                                   torch_dtype=dtype).to(device)

    # Prepare watermark information
    message = np.zeros(256, dtype=int)

    wm, H, G = ldpc_encode(message=message, batch_size=batch_size, redundancy=redundancy, CR=CR)

    # Sampling loop
    for index, img in islice(enumerate(dataLoader), start_index, None):
        img = img.to(device)

        # VAE Encoding (Img to Latent)
        vae = pipe.vae
        latents = img_to_latents(img, vae)

        # Inversion to recover noise
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
        inv_latents, _ = pipe(prompt=negative_prompt, negative_prompt=negative_prompt, guidance_scale=1,
                              width=img.shape[-1], height=img.shape[-2], output_type='latent', return_dict=False,
                              num_inference_steps=inverse_step, latents=latents)

        # Parse watermark from noise
        final_tensor = latentsToWatermark(wm.shape[1], inv_latents)
        visual_error_bits(final_tensor, redundancy, visual_img_path, batch_size, index)


def sdedit_batch_process(input_dir, output_dir, t=0.3, batch_size=8, steps=50, sampler="dpm++",
                         model_id="CompVis/stable-diffusion-v1-4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)

    # Switch Scheduler (Sampler)
    if sampler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpm++":
        # Currently the most recommended general-purpose sampler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif sampler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # print(pipe.scheduler)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image paths
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 2. Batch Processing
    for i in tqdm(range(0, len(all_files), batch_size), desc="Processing Batches"):
        batch_filenames = all_files[i: i + batch_size]
        batch_imgs = []

        # Pre-process images in this batch
        for fname in batch_filenames:
            raw_img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            # All images in a batch must have the same dimensions; resizing to 512x512 here.
            # If dimensions differ, Tensors cannot be stacked.
            batch_imgs.append(raw_img.resize((512, 512)))

        # 3. Execute SDEdit batch generation
        with torch.inference_mode():
            # The 'image' parameter in pipe can directly accept a list of PIL Images
            outputs = pipe(
                prompt=["masterpiece, high quality"] * len(batch_imgs),
                image=batch_imgs,
                strength=t,
                num_inference_steps=steps,  # Determines total diffusion steps
                guidance_scale=7.5
            ).images

        # 4. Save results in batch
        for j, output_img in enumerate(outputs):
            output_img.save(os.path.join(output_dir, batch_filenames[j]))
