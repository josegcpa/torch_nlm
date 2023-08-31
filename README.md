# Memory efficient non-local means on PyTorch

## Context

Non-local means takes a long time compute as it is quadratic for the number of pixels in an image. For simple images this is OK. For larger or three-dimensional (common in medical imaging) images this is impeditive for its application. Here, I introduce a PyTorch-based solution which uses convolutions to extract neighbours (non-local means here does not use the complete image but rather a neighbourhood with $n*n$ pixels or $n*n*n$ voxels) and calculates the non-local means average. By porting this to PyTorch we can make easy use of the very efficient GPU parallelization and speed up what oftentimes is a very time consuming algorithm.

### Brief explanation of non-local means (NLM)

For an image $I \in \mathbb{R}^{h \times w}$ consider pixel $I_{i,j}$ with coordinates $i,j$. 

To obtain the non-local mean of this pixel:

$$\frac{1}{W}\sum{}^{h,w}_{a,b=1} w(I_{i,j},I_{a,b}) * I_{a,b}$$

where $w(I_{i,j},I_{a,b})$ is the weight of pixel $I_{a,b}$ given $I_{i,j}$ and $W=\sum{}^{h,w}_{a,b=1} w(I_{i,j},I_{a,b})$. 

In other words, the non-local means of a given pixel is the weighted average of all pixels. Weights, here, are calculated as the $w(I_{i,j},I_{a,b}) = \exp(- \frac{(I_{i,j} - I_{a,b})^2}{h^2})$, where $h$ is a constant (akin to a standard deviation). To make computation tractable, rather than looping over all pixels, a simple solution is to restrict the neighbourhood to a small square surrounding the image - this is the solution used here.

## Implementation details

This code was optimized for speed. Three main functions are provided here - `apply_nonlocal_means_2d`, `apply_windowed_nonlocal_means_2d` and `apply_nonlocal_means_2d_mem_efficient`. 

### `apply_nonlocal_means_2d`

Retrieves all neighbours as a large tensor and calculates the NLM of the image. 

**Problems:** large neighbourhoods will lead to OOM

### `apply_windowed_nonlocal_means_2d`

Does the same as `apply_nonlocal_means_2d` but uses strided patches to do this, thus reducing memory requirements.

**Problems:** leads to visible striding patch artifacts 

### `apply_nonlocal_means_2d_mem_efficient`

Does the same as `apply_nonlocal_means_2d` but loops over sets of neighbourhoods to calculate weights. $W$ is updated as an accummulator.

## Generalising to 3D

The good aspect of this is that it requires very little effort to generalise these functions to 3D. These are made available with the same names as above but replacing `2d` with `3d`.

## Expected runtime

For a large image such as `assets/threegorges-1024x1127.jpg` ([source](https://www.lightstalking.com/wp-content/uploads/threegorges-1024x1127.jpg); size: 1024x1127), `apply_nonlocal_means_2d_mem_efficient` takes ~3-5 seconds with a neighbourhood with 51x51 pixels when running on GPU.

Example below (obtained by running `python test.py assets/threegorges-1024x1127.jpg`): 

![](assets/test_output.jpg)

* First panel: original image
* Second panel: original image + noise
* Third panel: original image + noise + NLM
* Fourth panel: difference between original image and original image + noise + NLM

**Note on benchmarking:** while 2d benchmarks are reasonable, 3d benchmarks will take a lot of time because of `scikit-image`'s implementation. Expect times of ~4,000 seconds for a $256 \times 256 \times 256$ images with a neighbourhood size of 17 (`nlm_torch` ran in ~70-80 seconds 😊).