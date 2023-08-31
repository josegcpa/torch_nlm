import os
import sys
import time
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from pathlib import Path
from PIL import Image

image_path = sys.argv[1]
output_path = os.path.join(str(Path(image_path).parent),"test_output_skimage.jpg")

dev = "cuda:1"
sd = 0.05
kernel_size = 11
salt_prob = 0.05
pepper_prob = 0.05

image = np.array(Image.open(image_path))
image = image.mean(-1) / 255.
noise = np.random.randn(*image.shape)
image_with_noise = image + noise
image_with_noise = np.where(
    np.random.rand(*image_with_noise.shape) < salt_prob,
    1,image_with_noise)
image_with_noise = np.where(
    np.random.rand(*image_with_noise.shape) < pepper_prob,
    0,image_with_noise)
image_with_noise = np.clip(image_with_noise,0,1)

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1)

sigma_est = np.mean(estimate_sigma(image_with_noise, channel_axis=-1))

a = time.time()
denoise_fast = denoise_nl_means(image_with_noise[:,:,None], 
                                h=0.8 * sigma_est, fast_mode=True,
                                **patch_kw)
b = time.time()
print("Ellapsed time:",b-a)

output_image = [
    image,
    image_with_noise,
    denoise_fast,
    image - denoise_fast]

Image.fromarray(
    np.uint8(
        np.concatenate(output_image,1) * 255)
        ).save(output_path)