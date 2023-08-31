import os
import sys
import time
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import torch
from torch_nlm import nlm3d

ndim = 3
image_sizes = [[32 for _ in range(ndim)],
               [64 for _ in range(ndim)],
               [128 for _ in range(ndim)],
               [256 for _ in range(ndim)]]
neighbourhood_sizes = [5,11,17]

dev = "cuda:1"
sd = 0.05
kernel_size = 11
salt_prob = 0.05
pepper_prob = 0.05
sub_filter_size = 16
i = 0
for neighbourhood_size in neighbourhood_sizes:
    for image_size in image_sizes:
        image = np.random.rand(*image_size)
        noise = np.random.rand(*image.shape)
        image_with_noise = image + noise
        image_with_noise = np.where(
            np.random.rand(*image_with_noise.shape) < salt_prob,
            1,image_with_noise)
        image_with_noise = np.where(
            np.random.rand(*image_with_noise.shape) < pepper_prob,
            0,image_with_noise)
        image_with_noise = np.clip(image_with_noise,0,1)

        patch_kw = dict(patch_size=5,
                        patch_distance=neighbourhood_size // 2,
                        channel_axis=-1)

        a = time.time()
        sigma_est = np.mean(estimate_sigma(image_with_noise, channel_axis=-1))
        denoise_fast = denoise_nl_means(image_with_noise[:,:,None], 
                                        h=0.8 * sigma_est, fast_mode=True,
                                        **patch_kw)
        b = time.time()
        base_str = f"{image_size[0]}_{image_size[1]}_{image_size[2]},{neighbourhood_size}"
        print(base_str + f",skimage,{b-a}")

        patch_kw_torch = dict()

        image_with_noise_torch = torch.as_tensor(image_with_noise).to(dev)
        if i == 0:
            # wake pytorch up
            denoise_fast = nlm3d(
                image_with_noise_torch,kernel_size=neighbourhood_size,kernel_size_mean=5,
                sub_filter_size=sub_filter_size)
        a = time.time()
        denoise_fast = nlm3d(
            image_with_noise_torch,kernel_size=neighbourhood_size,kernel_size_mean=5,
            sub_filter_size=sub_filter_size)
        b = time.time()
        print(base_str + f",torch,{b-a}")

        torch.cuda.empty_cache()
        i += 1
