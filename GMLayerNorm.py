import torch
import torch.nn as nn
from typing import Type, Union
from math import prod
ArrayLike = Union[torch.Tensor, float]

class GMLayerNorm(nn.Module):
    """
    Gaussian Mixture Layer Normalization\n
    Applies Gaussian Mixture Normalization over a mini-batch of inputs
    Statistics are computed over the last `ndim` dimensions. 
    """
    
    def __init__(self,
                 normalized_size: ArrayLike,
                 bias: bool = True,
                 ):
        super(GMLayerNorm, self).__init__()
        self.normal_dist = torch.distributions.Normal(0, 1)
        self.norm_size = tuple(normalized_size) if type(normalized_size) is not int else (normalized_size,)
        self.ndim = len(self.norm_size)
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size)) if bias else None
    
    def normalize2d(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = x.mean(dim=-1), x.std(dim=-1)
        start_mean = torch.stack((mean-1.2*std, mean, mean+1.2*std), dim=1)
        z: torch.Tensor = self.normal_dist.log_prob((x[:, :, None] - start_mean[:, None, :])/std[:, None, None]*3).exp() / std[:, None, None]
        z /= z.sum(dim=-1, keepdim=True)
        n_k: torch.Tensor = z.sum(dim=-2)
        gmm_means: torch.Tensor = torch.sum(x[:, :, None] * z, dim=-2) / n_k
        gmm_std: torch.Tensor = torch.sqrt(torch.sum(z * torch.square(x[:, :, None]), dim=-2) / n_k - torch.square(gmm_means))
        gmm_p: torch.Tensor = n_k / z.shape[-2]
        cdf: torch.Tensor = torch.sum(gmm_p[:, None, :] * self.normal_dist.cdf((x[:, :, None] - gmm_means[:, None, :])/gmm_std[:, None, :]), dim=-1)
        x = self.normal_dist.icdf(cdf)
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian Mixture Normalization to the input tensor"""
        self.extra_size = x.shape[:-self.ndim]
        slice_shape = [prod(self.extra_size), prod(self.norm_size)]
        x = x.reshape(slice_shape)
        x = self.normalize2d(x)
        x = x.reshape(self.extra_size + self.norm_size)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x