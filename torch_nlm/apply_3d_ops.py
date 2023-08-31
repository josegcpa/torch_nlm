"""
Functions for 3d NLM using torch.
"""

__author__ = "José Guilherme de Almeida"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "José Guilherme de Almeida"
__email__ = "jose.almeida@research.fchampalimaud.org"

import torch
import einops
import torch.nn.functional as F
from itertools import product
from tqdm import tqdm
from typing import Tuple
from .base_ops import (unsqueeze_tensor_at_dim,
                       apply_mean_filter,
                       get_neighbours,
                       non_local_means_loop_index)

def apply_nonlocal_means_3d(X:torch.Tensor,
                            kernel_size:int=3,
                            std:float=1,
                            kernel_size_mean=3):
    """
    Calculates non-local means for an input X with 3 dimensions.

    Args:
        X (torch.Tensor): input tensor with shape [h,w,d].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
        filtering.

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    ndim = 3
    # include batch and channel dimensions
    X = unsqueeze_tensor_at_dim(X,ndim+2)
    padding = (
        kernel_size // 2, kernel_size // 2, 
        kernel_size // 2, kernel_size // 2,
        kernel_size // 2, kernel_size // 2)
    # apply mean filter
    X = apply_mean_filter(X,kernel_size_mean,ndim)
    # retrieve neighbourhood
    neighbours_X = get_neighbours(
        F.pad(X,padding),kernel_size=kernel_size,ndim=ndim)
    distances = torch.sqrt(torch.square(X - neighbours_X))
    weights = torch.exp(- distances / std**2)
    weights = weights / weights.sum(1,keepdim=True)
    output = (neighbours_X * weights).sum(1).squeeze(0)
    return output

def apply_windowed_nonlocal_means_3d(X:torch.Tensor,
                                     kernel_size:int=3,
                                     std:int=1,
                                     kernel_size_mean=3,
                                     window_size:Tuple[int,int,int]=[128,128,128],
                                     strides:Tuple[int,int,int]=[64,64,64]):
    """
    Calculates non-local means for an input X with 3 dimensions by calculating 
    a local (windowed) non-local means. Leads to artefacts because of this - 
    not really good as an implementation, kept mostly as a curiosity.

    Args:
        X (torch.Tensor): input tensor with shape [h,w,d].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
            filtering.
        window_size (Tuple[int,int,int], optional): size of window. Defaults to
            [128,128,128].
        strides (Tuple[int,int,int], optional): size of window. Defaults to
            [64,64,64].

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    ndim = 3
    output = torch.zeros_like(X)
    denominator = torch.zeros_like(X)
    # include batch and channel dimensions
    X = unsqueeze_tensor_at_dim(X,ndim+2)
    sh = X.shape
    b,c,h,w,d = sh[0],sh[1],sh[2],sh[3],sh[4]
    padding = (
        kernel_size // 2, kernel_size // 2, 
        kernel_size // 2, kernel_size // 2,
        kernel_size // 2, kernel_size // 2)
    X = apply_mean_filter(X,kernel_size_mean,ndim)
    X = F.pad(X,padding)
    neighbours_X = get_neighbours(X,kernel_size=kernel_size,ndim=ndim)
    X = X[:,:,
          padding[0]:(X.shape[2]-padding[0]),
          padding[1]:(X.shape[3]-padding[1]),
          padding[2]:(X.shape[3]-padding[2])]
    
    all_ij = list(product(range(0,h,strides[0]),
                          range(0,w,strides[1]),
                          range(0,d,strides[2])))

    for i,j,k in tqdm(all_ij):
        i_1,i_2 = i,i+window_size[0]
        j_1,j_2 = j,j+window_size[1]
        k_1,k_2 = k,k+window_size[2]
        if i_2 > h:
            i_1,i_2 = h - window_size[0], h
        if j_2 > w:
            j_1,j_2 = w - window_size[1], w
        if k_2 > w:
            k_1,k_2 = d - window_size[2], d
        # reshape X to calculate distances
        sub_X = X[:,:,i_1:i_2,j_1:j_2,k_1:k_2]
        sub_neighbours = neighbours_X[:,:,i_1:i_2,j_1:j_2,k_1:k_2]
        reshaped_neighbours_X = einops.rearrange(
            sub_neighbours,"b c h w d -> b (h w d) c")
        # calculate distances
        neighbour_dists = torch.cdist(
            reshaped_neighbours_X,reshaped_neighbours_X)
        # calculate weights from distances
        weights = torch.exp(-neighbour_dists / (std**2))
        # calculate the new values 
        flat_X = einops.rearrange(sub_X,"b c h w d -> b (h w d) c")
        weighted_X = (weights @ flat_X).squeeze(-1)
        output[i_1:i_2,j_1:j_2,k_1:k_2] += einops.rearrange(
            weighted_X,"b (h w d) -> b h w d",
            b=b,h=window_size[0],w=window_size[1],d=window_size[2]).squeeze(0)
        denominator[i_1:i_2,j_1:j_2,k_1:k_2] += einops.rearrange(
            weights.sum(-1),"b (h w d) -> b h w d",
            b=b,h=window_size[0],w=window_size[1],d=window_size[2]).squeeze(0)
    output = output / denominator
    return output

def apply_nonlocal_means_3d_mem_efficient(X:torch.Tensor,
                                          kernel_size:int=3,
                                          std:int=1,
                                          kernel_size_mean=3,
                                          sub_filter_size:int=256):
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. Most of the heavy lifting is performed by
    non_local_means_loop.

    Args:
        X (torch.Tensor): input tensor with shape [h,w,d].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
            filtering.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non local means).

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    ndim = 3
    # include batch and channel dimensions
    X = unsqueeze_tensor_at_dim(X,ndim+2)
    # apply mean filter
    X = apply_mean_filter(X,kernel_size_mean,ndim)
    # retrieve neighbourhood
    output = non_local_means_loop_index(
        X,kernel_size=kernel_size,std=std,
        ndim=ndim,sub_filter_size=sub_filter_size).squeeze(0).squeeze(0)
    return output

nlm3d = apply_nonlocal_means_3d_mem_efficient