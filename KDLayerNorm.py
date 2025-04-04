# The normalization module to be used with the Torch.NN model framework
import torch
import torch.nn as nn
import numpy as np
from typing import Type, Union
from torch.nn import functional as F
from math import prod
ArrayLike = Union[torch.Tensor, np.ndarray, float]
from abc import ABC, abstractmethod


# bandwidth heuristics
class BandwidthHeuristic:
    @staticmethod
    def scott(x: torch.Tensor) -> torch.Tensor:
        return 1.059 * x.std() * x.numel() ** (-1/5)
    @staticmethod
    def silverman(x: torch.Tensor) -> torch.Tensor:
        return 0.9 * x.std() * x.numel() ** (-1/5)
    @staticmethod
    def scott2d(x: torch.Tensor) -> torch.Tensor:
        return 1.059 * x.std(dim=-1) * x.shape[-1] ** (-1/5)
    @staticmethod
    def silverman2d(x: torch.Tensor) -> torch.Tensor:
        return 0.9 * x.std(dim=-1) * x.shape[-1] ** (-1/5)
    
class DensityKernel(ABC):
    """Abstract base class for kernel functions used in Kernel Density Estimation"""
    
    @abstractmethod
    def evaluate(x: torch.Tensor) -> torch.Tensor:
        """Evaluate the kernel at all points in x"""
        pass

    @abstractmethod
    def cdf(x: torch.Tensor) -> torch.Tensor:
        """Evaluate the CDF of the kernel at all points in x"""
        pass

    @abstractmethod
    def ppf(x: torch.Tensor) -> torch.Tensor:
        """Evaluate the inverse CDF of the kernel at all points in x"""
        pass

class GaussianKernel(DensityKernel):
    """Gaussian kernel for density estimation"""
    
    @staticmethod
    def evaluate(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * torch.square(x)) / (np.sqrt(2 * np.pi))
    
    @staticmethod
    def cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))
    
    @staticmethod
    def ppf(x: torch.Tensor) -> torch.Tensor:
        return torch.erfinv(2*x - 1) * np.sqrt(2)

class KernelDensityEstimator:
    """Implements Kernel Density Estimation (KDE) with a given kernel"""

    def __init__(self, data: torch.Tensor, kernel: Type[DensityKernel] = GaussianKernel, bandwidth: ArrayLike = None, bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman):
        self.data = data
        self.kernel = kernel
        if data is not None:
            self.bandwidth = bandwidth if bandwidth is not None else bandwidth_heuristic(data)
    
    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the density at points x using the given kernel"""
        scaled_x = (x[:,np.newaxis] - self.data[np.newaxis,:]) / self.bandwidth
        return torch.mean(self.kernel.evaluate(scaled_x), dim=1) / self.bandwidth
    
    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the CDF at points x using the given kernel"""
        scaled_x = (x[:,np.newaxis] - self.data[np.newaxis,:]) / self.bandwidth
        return torch.mean(self.kernel.cdf(scaled_x), dim=1)
    
    def normalize(self) -> torch.Tensor:
        """Normalize the data using the estimated CDF"""
        cdf = self.cdf(self.data)
        return self.kernel.ppf(cdf)
    
    def normalize_data(self, data: torch.Tensor, bandwidth: ArrayLike = None, bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman) -> torch.Tensor:
        """Normalize the given data using the estimated CDF"""
        self.bandwidth = bandwidth if bandwidth is not None else bandwidth_heuristic(data)
        self.data = data
        cdf = self.cdf(data)
        return self.kernel.ppf(cdf)
    
    def cdf2d(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the CDF at points x using the given kernel"""
        scaled_x = (x[:,:,None] - self.data[:,None,:]) / self.bandwidth[:,None,None]
        return torch.mean(self.kernel.cdf(scaled_x), dim=2)

    def cdf2d_data(self, x: torch.Tensor, data: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
        """Estimate the CDF at points x using the given kernel"""
        scaled_x = (x[:,:,None] - data[:,None,:]) / bandwidth[:,None,None]
        #densities = []
        #for x_batch in x.split(1000):
        #    scaled_x = (x_batch[:,:,None] - data[:,None,:]) / bandwidth[:,None,None]
        #    densities.append()
        #print(f"Tensor size: {scaled_x.element_size() * scaled_x.nelement()} Bytes")
        return torch.mean(self.kernel.cdf(scaled_x), dim=2)
    
    def normalize_data2d(self, data: torch.Tensor, bandwidth: ArrayLike = None, bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman2d) -> torch.Tensor:
        #used to be bandwidth heuristic
        #used to be self.data = data
        #cdf = self.cdf2d(data)
        return self.kernel.ppf(self.cdf2d_data(data, data, bandwidth if bandwidth is not None else bandwidth_heuristic(data)))

class KDLayerNorm(nn.Module):
    """
    Kernel Density Layer Normalization\n
    Applies Kernel Density Normalization over a mini-batch of inputs
    Statistics are computed over the last `ndim` dimensions. 
    """
    
    def __init__(self,
                 normalized_size: ArrayLike,
                 bias: bool = True,
                 kernel: Type[DensityKernel] = GaussianKernel,
                 bandwidth: ArrayLike = None,
                 bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman2d,
                 ):
        super(KDLayerNorm, self).__init__()
        self.kde = KernelDensityEstimator(None, kernel, bandwidth, bandwidth_heuristic)
        self.explicit_bandwidth = bandwidth
        self.bandwidth_heuristic = bandwidth_heuristic
        self.norm_size = tuple(normalized_size) if type(normalized_size) is not int else (normalized_size,)
        self.ndim = len(self.norm_size)
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Kernel Density Normalization to the input tensor"""
        extra_size = x.shape[:-self.ndim]
        slice_shape = [prod(extra_size), prod(self.norm_size)]
        x = x.reshape(slice_shape)
        x = self.kde.normalize_data2d(x, self.explicit_bandwidth, self.bandwidth_heuristic)
        x = x.reshape(extra_size + self.norm_size)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x
    
class KernelDensityEstimatorSmpl:
    """Implements Sampling Kernel Density Estimation (Smpl-KDE) with a given kernel"""

    def __init__(self,
                 data: torch.Tensor,
                 kernel: Type[DensityKernel] = GaussianKernel,
                 bandwidth: ArrayLike = None,
                 bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman,
                 sample_size: int = 32,
                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                 ):
        self.data = data
        self.kernel = kernel
        self.sample_size = sample_size
        self.device = device
        if data is not None:
            self.bandwidth = bandwidth if bandwidth is not None else bandwidth_heuristic(data)
    
    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the density at points x using the given kernel"""
        scaled_x = (x[:,np.newaxis] - self.data[np.newaxis,:]) / self.bandwidth
        return torch.mean(self.kernel.evaluate(scaled_x), dim=1) / self.bandwidth
    
    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the CDF at points x using the given kernel"""
        scaled_x = (x[:,np.newaxis] - self.data[np.newaxis,:]) / self.bandwidth
        return torch.mean(self.kernel.cdf(scaled_x), dim=1)
    
    def normalize(self) -> torch.Tensor:
        """Normalize the data using the estimated CDF"""
        cdf = self.cdf(self.data)
        return self.kernel.ppf(cdf)
    
    def normalize_data(self, data: torch.Tensor, bandwidth: ArrayLike = None, bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman) -> torch.Tensor:
        """Normalize the given data using the estimated CDF"""
        self.bandwidth = bandwidth if bandwidth is not None else bandwidth_heuristic(data)
        self.data = data
        cdf = self.cdf(data)
        return self.kernel.ppf(cdf)
    
    def cdf2d(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the CDF at points x using the given kernel"""
        scaled_x = (x[:,:,None] - self.data[:,None,:]) / self.bandwidth[:,None,None]
        return torch.mean(self.kernel.cdf(scaled_x), dim=2)
    
    def normalize_data2d(self, data: torch.Tensor, bandwidth: ArrayLike = None, bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman2d) -> torch.Tensor:
        """Normalize the given data using the estimated CDF"""
        data_sample = data[:, torch.randint(data.shape[-1], (self.sample_size,))]
        self.bandwidth = bandwidth if bandwidth is not None else bandwidth_heuristic(data_sample)
        self.data = data_sample
        cdf = self.cdf2d(data)
        return self.kernel.ppf(cdf)

class KDLayerNormSmpl(nn.Module):
    """
    Sampling Kernel Density Layer Normalization\n
    Applies Sampling Kernel Density Normalization over a mini-batch of inputs
    Statistics are computed over the last `ndim` dimensions. 
    """
    
    def __init__(self,
                 normalized_size: ArrayLike,
                 bias: bool = True,
                 kernel: Type[DensityKernel] = GaussianKernel,
                 bandwidth: ArrayLike = None,
                 bandwidth_heuristic: Type[BandwidthHeuristic] = BandwidthHeuristic.silverman2d,
                 sample_size: int = 32,
                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                 ):
        super(KDLayerNormSmpl, self).__init__()
        self.kde = KernelDensityEstimatorSmpl(None, kernel, bandwidth, bandwidth_heuristic, sample_size, device)
        self.explicit_bandwidth = bandwidth
        self.bandwidth_heuristic = bandwidth_heuristic
        self.norm_size = tuple(normalized_size) if type(normalized_size) is not int else (normalized_size,)
        self.ndim = len(self.norm_size)
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sampling Kernel Density Normalization to the input tensor"""
        self.extra_size = x.shape[:-self.ndim]
        slice_shape = [prod(self.extra_size), prod(self.norm_size)]
        x = x.reshape(slice_shape)
        x = self.kde.normalize_data2d(x, self.explicit_bandwidth, self.bandwidth_heuristic)
        x = x.reshape(self.extra_size + self.norm_size)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x