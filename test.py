import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch_nlm import nlm2d

image_path = sys.argv[1]
output_path = os.path.join(str(Path(image_path).parent),"test_output.jpg")

dev = "cuda:1"
sd = 0.05
kernel_size = 51
salt_prob = 0.05
pepper_prob = 0.05

image = torch.as_tensor(np.array(Image.open(image_path)),device=dev)
image = image.float().mean(-1) / 255.
noise = torch.randn(size=image.shape).to(image) * sd
image_with_noise = image + noise
image_with_noise = torch.where(
    torch.rand_like(image_with_noise) < salt_prob,
    1,image_with_noise)
image_with_noise = torch.where(
    torch.rand_like(image_with_noise) < pepper_prob,
    0,image_with_noise)
image_with_noise = torch.clamp(image_with_noise,0,1)

a = time.time()
output = apply_nonlocal_means_2d_mem_efficient(
    image_with_noise,kernel_size=kernel_size,std=1.0,sub_filter_size=32)
b = time.time()
print("Ellapsed time:",b-a)

output_image = [
    image,
    image_with_noise,
    output,
    image - output]

Image.fromarray(
    np.uint8(
        torch.cat(output_image,1).cpu().numpy() * 255)
        ).save(output_path)