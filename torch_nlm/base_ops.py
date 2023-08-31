"""
Generic operations for torch_nlm.
"""

__author__ = "José Guilherme de Almeida"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "José Guilherme de Almeida"
__email__ = "jose.almeida@research.fchampalimaud.org"

import numpy as np
import torch
import torch.nn.functional as F
from itertools import product 
from tqdm import tqdm
from typing import Tuple,List

def get_gaussian_kernel(kernel_size:int=5, sigma:float=1.,ndim:int=2)->np.ndarray:
    """Creates gaussian kernel with side length kernel_size and a standard 
    deviation of sigma.

    Based on: https://stackoverflow.com/a/43346070

    Args:
        kernel_size (int, optional): size of kernel. Defaults to 5.
        sigma (float, optional): sigma for the normal distribution. Defaults 
            to 1.
        ndim (int, optional): number of dimensions in output kernel.

    Returns:
        np.ndarray: Gaussian filter kernel.
    """
    ax = torch.linspace(-(kernel_size - 1) / 2., 
                        (kernel_size - 1) / 2.,
                        kernel_size)
    gauss = torch.exp(-0.5 * np.square(ax) / np.square(sigma))
    if ndim == 1:
        kernel = gauss
    elif ndim == 2:
        kernel = torch.outer(gauss, gauss)
    elif ndim == 3:
        kernel = gauss[None,None,:] * gauss[None,:,None] * gauss[:,None,None]
    return kernel / torch.sum(kernel)

def unsqueeze_tensor_at_dim(X:torch.Tensor,ndim:int,dim:int=0)->torch.Tensor:
    """
    Adds dimensions as necessary.

    Args:
        X (torch.Tensor): tensor.
        ndim (int): number of output dimensions.
        dim (int): dimension which will be unsqueezed. Defaults to 0.

    Returns:
        torch.Tensor: unqueezed tensor.
    """
    sh = len(X.shape)
    diff = ndim - sh
    if diff > 0:
        for _ in range(diff):
            X = X.unsqueeze(0)
    return X

def make_neighbours_kernel(kernel_size:int=3,ndim:int=2)->torch.Tensor:
    """
    Make convolutional kernel that extracts neighbours within a kernel_size
        neighbourhood (each filter is 1 for the corresponding neighbour and
        0 otherwise). 

    Args:
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.

    Returns:
        torch.Tensor: convolutional kernel for neighbourhood extraction.
    """
    K = kernel_size ** ndim
    filter = torch.zeros([K,1,*[kernel_size for _ in range(ndim)]])
    generators = [range(kernel_size) for _ in range(ndim)]
    for i,coord in enumerate(product(*generators)):
        if ndim == 2:
            filter[i,:,coord[0],coord[1]] = 1
        elif ndim == 3:
            filter[i,:,coord[0],coord[1],coord[2]] = 1
    return filter

def get_neighbours(X:torch.Tensor,
                   kernel_size:int=3,
                   ndim:int=2)->torch.Tensor:
    """
    Retrieves neighbours in an image and stores them in the channel dimension.
    Expects the input to be padded.

    Args:
        X (torch.Tensor): 4-D or 5-D (batched) tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.

    Returns:
        torch.Tensor: Gaussian filter-normalised X.
    """
    filter = make_neighbours_kernel(kernel_size,ndim).to(X)
    if ndim == 2:
        X = F.conv2d(X,filter,padding=0)
    elif ndim == 3:
        X = F.conv3d(X,filter,padding=0)
    else:
        raise NotImplementedError("ndim must be 2 or 3.")
    return X

def apply_gaussian_filter(X:torch.Tensor,
                          kernel_size:int=3,
                          ndim:int=2,
                          sigma:float=1.0)->torch.Tensor:
    """
    Simple function to apply Gaussian filter.

    Args:
        X (torch.Tensor): input tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.
        sigma (float, optional): standard deviation for the filter. Defaults to
            1.0.

    Returns:
        torch.Tensor: mean filter-normalised X.
    """
    X = unsqueeze_tensor_at_dim(X,ndim=ndim+2)
    n_channels = X.shape[1]
    gaussian_kernel = get_gaussian_kernel(kernel_size,sigma,ndim)
    gaussian_kernel = unsqueeze_tensor_at_dim(gaussian_kernel,ndim=ndim)
    gaussian_kernel = torch.cat([gaussian_kernel for _ in range(n_channels)],0)
    gaussian_kernel = torch.cat([gaussian_kernel for _ in range(n_channels)],1)
    if ndim == 2:
        X = F.conv2d(X,gaussian_kernel,bias=0,padding=kernel_size//2)
    elif ndim == 3:
        X = F.conv3d(X,gaussian_kernel,bias=0,padding=kernel_size//2)
    else:
        raise NotImplementedError("ndim must be 2 or 3.") 
    return X

def apply_mean_filter(X:torch.Tensor,
                      kernel_size:int=3,
                      ndim:int=2):
    """
    Simple function to apply mean filter.

    Args:
        X (torch.Tensor): input tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.
        sigma (float, optional): standard deviation for the filter. Defaults to
            1.0.

    Returns:
        torch.Tensor: _description_
    """
    filter = torch.ones([kernel_size for _ in range(ndim)]).to(X)
    filter = filter / filter.sum()
    filter = unsqueeze_tensor_at_dim(filter,ndim+2)
    pad = kernel_size // 2
    padding = tuple([pad for _ in range(ndim * 2)])
    X = F.pad(X,padding,mode="reflect")
    if ndim == 2:
        X = F.conv2d(X,filter,padding=0)
    if ndim == 3:
        X = F.conv3d(X,filter,padding=0)
    return X

def array_chunk(arr:np.ndarray,chunk_size:int):
    for i in range(0,arr.shape[0],chunk_size):
        yield arr[i:i+chunk_size]

def non_local_means_loop_index(X:torch.Tensor,
                               kernel_size:int=3,
                               ndim:int=2,
                               sub_filter_size:int=1,
                               std:float=1.0)->torch.Tensor:
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. This avoids storing at any given stage a large 
    number of neighbours, making the calculation of larger neighbourhoods 
    possible without running the risk of OOM. Here, neigbhours are selected 
    using simple indexing.

    Args:
        X (torch.Tensor): input tensor.
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Must be 2 or 3. Defaults to
            2.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non local means).
        std (float, optional): standard deviation for weights. Defaults to 1.0.

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    def cat_idxs(idxs:List[torch.Tensor]):
        for i in range(len(idxs)):
            idxs[i] = torch.cat(idxs[i])
        return idxs

    def preprocess_idxs(idxs:List[torch.Tensor],
                        original_idxs:List[torch.Tensor],
                        sizes:List[int]):
        for i in range(len(idxs)):
            d = torch.abs(original_idxs[i] - idxs[i])
            idxs[i] = torch.where(
                idxs[i] < 0, 
                original_idxs[i] + d,
                idxs[i])
            idxs[i] = torch.where(
                idxs[i] > sizes[i] - 1, 
                original_idxs[i] - d,
                idxs[i])
        return idxs
    
    def calculate_weights_2d(X:torch.Tensor,
                             idxs:List[torch.Tensor],
                             std_2:torch.Tensor)->Tuple[torch.Tensor,
                                                        torch.Tensor]:
        n = len(idxs[0])
        idxs = cat_idxs(idxs)
        neighbours = X[:,:,idxs[0],idxs[1]].reshape(
            1,n,X.shape[2],X.shape[3])
        weights = torch.square(
            X - neighbours).negative().divide(std_2).exp()
        return weights, neighbours

    def calculate_weights_3d(X:torch.Tensor,
                             idxs:List[torch.Tensor],
                             std_2:torch.Tensor)->Tuple[torch.Tensor,
                                                        torch.Tensor]:
        n = len(idxs[0])
        idxs = cat_idxs(idxs)
        neighbours = X[:,:,idxs[0],idxs[1],idxs[2]].reshape(
            1,n,X.shape[2],X.shape[3],X.shape[4])
        weights = torch.square(
            X - neighbours).negative().divide(std_2).exp()
        return weights, neighbours

    weights_sum = torch.zeros_like(X)
    output = torch.zeros_like(X)
    k2 = kernel_size // 2
    std_2 = torch.as_tensor(std**2).to(X)
    if ndim == 2:
        _,_,H,W = X.shape
        h,w = torch.where(X[0,0] == X[0,0])
        all_idxs = [[],[]]
        range_h = torch.arange(-k2,k2+1,dtype=torch.long)
        range_w = torch.arange(-k2,k2+1,dtype=torch.long)
        for i in range_h:
            for j in range_w:
                new_idxs = preprocess_idxs(
                    [h + i,w + j],[h,w],[H,W])
                all_idxs[0].append(new_idxs[0])
                all_idxs[1].append(new_idxs[1])
                if len(all_idxs[0]) >= sub_filter_size:
                    weights,neighbours = calculate_weights_2d(X,all_idxs,std_2)
                    weights_sum += torch.sum(weights,1)
                    output += weights.multiply(neighbours).sum(1)
                    all_idxs = [[],[]]
        if len(all_idxs[0]) >= 0:
            weights,neighbours = calculate_weights_2d(X,all_idxs,std_2)
            weights_sum += torch.sum(weights,1)
            output += weights.multiply(neighbours).sum(1)

    if ndim == 3:
        _,_,H,W,D = X.shape
        h,w,d = torch.where(X[0,0] == X[0,0])
        all_idxs = [[],[],[]]
        range_h = torch.arange(-k2,k2+1,dtype=torch.long)
        range_w = torch.arange(-k2,k2+1,dtype=torch.long)
        range_d = torch.arange(-k2,k2+1,dtype=torch.long)
        for i in range_h:
            for j in range_w:
                for k in range_d:
                    new_idxs = preprocess_idxs(
                        [h + i,w + j,d + k],[h,w,d],[H,W,D])
                    all_idxs[0].append(new_idxs[0])
                    all_idxs[1].append(new_idxs[1])
                    all_idxs[2].append(new_idxs[2])
                    if len(all_idxs[0]) >= sub_filter_size:
                        weights,neighbours = calculate_weights_3d(X,all_idxs,std_2)
                        weights_sum += torch.sum(weights,1)
                        output += weights.multiply(neighbours).sum(1)
                        all_idxs = [[],[],[]]
        if len(all_idxs[0]) >= 0:
            weights,neighbours = calculate_weights_3d(X,all_idxs,std_2)
            weights_sum += torch.sum(weights,1)
            output += weights.multiply(neighbours).sum(1)

    return X

def non_local_means_loop(X:torch.Tensor,
                         kernel_size:int=3,
                         ndim:int=2,
                         sub_filter_size:int=1,
                         std:float=1.0)->torch.Tensor:
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. This avoids storing at any given stage a large 
    number of neighbours, making the calculation of larger neighbourhoods 
    possible without running the risk of OOM. Here, neigbhours are selected 
    using convolutional filters.

    Args:
        X (torch.Tensor): input tensor.
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Must be 2 or 3. Defaults to
            2.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non local means).
        std (float, optional): standard deviation for weights. Defaults to 1.0.

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    filter = make_neighbours_kernel(kernel_size,ndim).to(X)
    pad_size = kernel_size // 2
    padding = tuple([pad_size for _ in range(4)])
    weights_sum = torch.zeros_like(X)
    output = torch.zeros_like(X)
    n_filters = filter.shape[0]
    if sub_filter_size > n_filters:
        blocks = [np.arange(n_filters,dtype=int)]
    else:
        blocks = list(
            array_chunk(np.arange(n_filters,dtype=int),sub_filter_size))
    padded_X = F.pad(X,padding)
    std_2 = torch.as_tensor(std**2).to(X)
    with torch.no_grad():
        for block in tqdm(blocks):
            neighbours = F.conv2d(
                padded_X,filter[block],padding=0)
            weights = torch.square(X - neighbours).negative().divide(std_2).exp()
            weights_sum += torch.sum(weights,1)
            output += weights.multiply(neighbours).sum(1)
    output = output / weights_sum
    return output
