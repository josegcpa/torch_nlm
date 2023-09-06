# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['torch_nlm']

package_data = \
{'': ['*']}

install_requires = \
['Pillow==10.0.0',
 'einops==0.6.1',
 'numpy==1.23.5',
 'torch==2.0.1',
 'tqdm==4.64.1']

setup_kwargs = {
    'name': 'nlm-torch',
    'version': '0.1.0',
    'description': 'PyTorch-based implementation of non-local means with GPU support.',
    'long_description': '# `torch_nlm` - memory efficient non-local means on PyTorch\n\n## Context\n\nNon-local means takes a long time compute as it is quadratic for the number of pixels in an image. For simple images this is OK. For larger or three-dimensional (common in medical imaging) images this is impeditive for its application. Here, I introduce a PyTorch-based solution which uses convolutions to extract neighbours (non-local means here does not use the complete image but rather a neighbourhood with $n*n$ pixels or $n*n*n$ voxels) and calculates the non-local means average. By porting this to PyTorch we can make easy use of the very efficient GPU parallelization and speed up what oftentimes is a very time consuming algorithm.\n\n## When should you use it?\n\n* **You want to run NLM for small images:** just use `scikit-image`\n* **You want to run NLM for bigger images AND you have a GPU:** use this\n* **You want to run NLM for relatively big images and you DO NOT have a GPU:** good luck\n\n## Benchmark\n\nI only benchmarked `torch_nlm` against `scikit-image` in 2d because the latter is prohibitively slow in 3d. Results below.\n\n![](assets/benchmark_results.png)\n\n### Brief explanation of non-local means (NLM)\n\nFor an image $I \\in \\mathbb{R}^{h \\times w}$ consider pixel $I_{i,j}$ with coordinates $i,j$. \n\nTo obtain the non-local mean of this pixel:\n\n$$\\frac{1}{W}\\sum{}^{h,w}_{a,b=1} w(I_{i,j},I_{a,b}) * I_{a,b}$$\n\nwhere $w(I_{i,j},I_{a,b})$ is the weight of pixel $I_{a,b}$ given $I_{i,j}$ and $W=\\sum{}^{h,w}_{a,b=1} w(I_{i,j},I_{a,b})$. \n\nIn other words, the non-local means of a given pixel is the weighted average of all pixels. Weights, here, are calculated as the $w(I_{i,j},I_{a,b}) = \\exp(- \\frac{(I_{i,j} - I_{a,b})^2}{h^2})$, where $h$ is a constant (akin to a standard deviation). To make computation tractable, rather than looping over all pixels, a simple solution is to restrict the neighbourhood to a small square surrounding the image - this is the solution used here.\n\n## Usage\n\nTo use this package all you have to do is clone and install this (a `pyproject.toml` is provided so that you can easily install this with [`poetry`](https://python-poetry.org/)). Alternatively, use `requirements.txt` with `pip` (i.e. `pip install -r requirements.txt`).\n\n**Installation with pip:** this is probably the version which will be the less painful to use: \n\n```bash\npip install torch_nlm\n```\n\nOr, if you already have all the dependencies:\n\n```bash\npip install torch_nlm --no-dependencies\n```\n\nTwo main functions are exported: `nlm2d` and `nlm3d`, which are aliases for the most efficient `torch`-based NLMM versions (`apply_nonlocal_means_2d_mem_efficient` and `apply_nonlocal_means_3d_mem_efficient`), respectively. So if you want to apply it to your favourite image and have a CUDA compatible GPU:\n\n```python\nimport torch # necessary for obvious reasons\nfrom torch_nlm import nlm2d\n\nimage = ... # here you define your image\n\nimage_torch = torch.as_tensor(image).to("cuda")\n\nimage_nlm = nlm2d(image_torch, # the image\n                  kernel_size=11, # neighbourhood size \n                  std=1.0, # the sigma \n                  kernel_size_mean=3, # the kernel used to compute the average pixel intensity\n                  sub_filter_size=32 # how many neighbourhoods are computed per iteration\n                  )\n```\n\n`sub_filter_size` is what allows large neighbourhoods - given that users may have relatively small GPU cards, they may opt for smaller `sub_filter_sizes` which will enable them to load much smaller sets of neighbourhoods for distance/weight calculations. You may want to run a few tests to figure out the best `sub_filter_size` before deploying this *en masse*.\n\n## Implementation details\n\nThis code was optimized for speed. Three main functions are provided here - `apply_nonlocal_means_2d`, `apply_windowed_nonlocal_means_2d` and `apply_nonlocal_means_2d_mem_efficient`. The first two are development versions, the latter is the one you should use (exposed as `nlm_2d`).\n\n### `apply_nonlocal_means_2d`\n\nRetrieves all neighbours as a large tensor and calculates the NLM of the image. \n\n**Problems:** large neighbourhoods will lead to OOM\n\n### `apply_windowed_nonlocal_means_2d`\n\nDoes the same as `apply_nonlocal_means_2d` but uses strided patches to do this, thus reducing memory requirements.\n\n**Problems:** leads to visible striding patch artifacts \n\n### `apply_nonlocal_means_2d_mem_efficient`\n\nDoes the same as `apply_nonlocal_means_2d` but loops over sets of neighbourhoods to calculate weights. $W$ is updated as an accummulator. This version requires defining a "batch size" analogue so that only a few neighbours are calculated at a time. This allows for massive neighbourhoods to be computed in a very parallel fashion.\n\n**Problems:** none for now! But time is a teacher to us all.\n\n## Generalising to 3D\n\nThe good aspect of this is that it requires very little effort to generalise these functions to 3D. These are made available with the same names as above but replacing `2d` with `3d`. The version you want to use is `nlm_3d`.\n\n## Expected runtime\n\nFor a large image such as `assets/threegorges-1024x1127.jpg` ([source](https://www.lightstalking.com/wp-content/uploads/threegorges-1024x1127.jpg); size: 1024x1127), `apply_nonlocal_means_2d_mem_efficient` takes ~3-5 seconds with a neighbourhood with 51x51 pixels when running on GPU.\n\nExample below (obtained by running `python test.py assets/threegorges-1024x1127.jpg`): \n\n![](assets/test_output.jpg)\n\n* First panel: original image\n* Second panel: original image + noise\n* Third panel: original image + noise + NLM\n* Fourth panel: difference between original image and original image + noise + NLM\n\n**Note on benchmarking:** while 2d benchmarks are reasonable, 3d benchmarks will take a lot of time because of `scikit-image`\'s implementation. Expect times of ~4,000 seconds for a $256 \\times 256 \\times 256$ images with a neighbourhood size of 17 (`torch_nlm` ran in ~70-80 seconds 😊). You will need `scikit-image` for benchmarking.',
    'author': 'josegcpa',
    'author_email': 'jose.almeida@research.fchampalimaud.pt',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.11,<4.0',
}


setup(**setup_kwargs)

# This setup.py was autogenerated using poetry.
