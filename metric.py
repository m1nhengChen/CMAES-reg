import torch

from sklearn.metrics.cluster import normalized_mutual_info_score as MI
from torch import nn
import torch.nn.functional as F
from patchify import patchify, unpatchify
import numpy as np
from einops import rearrange

# Calculate normalized cross-correlation
def cal_ncc(I, J, eps=1e-10):
    # Compute local sums via convolution
    
    B ,C, _, _ = I.shape
    I = I.reshape(B, C, -1)
    J= J.reshape(B, C, -1)
    I=I - I.mean(dim=-1,keepdim=True)
    J=J - J.mean(dim=-1,keepdim=True)
    # cross = (I - I.mean(dim=-1,keepdim=True)) * (J -  J.mean(dim=-1,keepdim=True))

    # cc = torch.sum(cross) / torch.sum(torch.sqrt(I_var * J_var + eps))
    cc = torch.sum(I * J, dim=-1) / (eps + torch.sqrt(torch.sum(I **2, dim=-1)) * torch.sqrt(torch.sum(J**2, dim=-1)))
    # cc = torch.clamp(cc, -1., 1.)
    # test = torch.mean(cc)
    return torch.mean(cc)


# Gradient-NCC loss
def gradncc(I, J, device='cuda', win=None, eps=1e-10):
    # Compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y

        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)

    return 1 - 0.5 * cal_ncc(Ix, Jx, eps) - 0.5 * cal_ncc(Iy, Jy, eps)


# MI loss
def mi(I, J):
    I = I.cpu().detach().numpy().flatten()
    J = J.cpu().detach().numpy().flatten()
    return 1 - MI(I, J)


# NCC loss
def ncc(I, J, device='cuda', win=None, eps=1e-10):
    return 1 - cal_ncc(I, J, eps)


# Cosine similarity
def cos_sim(a, b, device='cuda', win=None, eps=1e-10):
    return torch.sum(torch.multiply(a, b)) / ((torch.sum((a) ** 2) ** 0.5) * (torch.sum((b) ** 2)) ** 0.5 + eps)


# NCCL loss
def nccl(I, J, device='cuda', kernel_size=5, win=None, eps=1e-10):
    '''Normalized cross-correlation (NCCL) based on the LOG
    operator is obtained. The Laplacian image is obtained by convolution of the reference image
    and DRR image with the LOG operator. The zero-crossing point in the Laplacian image
    is no longer needed to obtain the image锟斤拷s detailed edge. However, two Laplacian images锟斤拷
    consistency is directly measured to use image edge and detail information effectively. This
    paper uses cosine similarity to measure the similarity between Laplacian images.'''
    
    # Compute filters
    with torch.no_grad():
        if kernel_size == 5:
            kernel_LoG = torch.Tensor([[[[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4], [-4, 0, 8, 0, -4],
                                         [-2, -4, -4, -4, -2]]]])
            kernel_LoG = torch.nn.Parameter(kernel_LoG, requires_grad=False)
            LoG = nn.Conv2d(1, 1, 5, 1, 1, bias=False)
        elif kernel_size == 9:
            kernel_LoG = torch.Tensor([[[[0, 1, 1, 2, 2, 2, 1, 1, 0],
                                         [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                         [1, 4, 5, 3, 0, 3, 5, 4, 1],
                                         [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                         [2, 5, 0, -24, -40, -24, 0, 5, 2],
                                         [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                         [1, 4, 5, 3, 0, 3, 4, 4, 1],
                                         [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                         [0, 1, 1, 2, 2, 2, 1, 1, 0]]]])
            kernel_LoG = torch.nn.Parameter(kernel_LoG, requires_grad=False)
            LoG = nn.Conv2d(1, 1, 9, 1, 1, bias=False)
        LoG.weight = kernel_LoG
        LoG = LoG.to(device)
    LoG_I = LoG(I)
    LoG_J = LoG(J)
    # Cosine_similarity
    return 1.5 - cal_ncc(I, J) - 0.5 * cos_sim(LoG_I, LoG_J)


# GD loss
def gradient_difference(I, J, s=1, device='cuda', win=None, eps=1e-10):
    # Compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y

        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)
    # Compute difference image
    if s != 1:
        Idx = Ix - s * Jx
        Idy = Iy - s * Jy
    else:
        Idx = Ix - Jx
        Idy = Iy - Jy
    # Compute variance of image
    N = torch.numel(Ix)
    Av = torch.sum((Ix - torch.mean(Ix)) ** 2) / N
    Ah = torch.sum((Iy - torch.mean(Iy)) ** 2) / N
    g = torch.sum(Av / (Av + (Idx) ** 2)) + torch.sum(Ah / (Ah + (Idy) ** 2))
    return 1 - 0.5 * g / N

class WeightedNormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images."""

    def __init__(self, patch_size=None, eps=1e-5):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(num_features=1, eps=eps)
        self.patch_size = patch_size

    def forward(self, x1, x2):
        if self.patch_size is not None:
            x1 = to_patches(x1, self.patch_size)
            x2 = to_patches(x2, self.patch_size)
        assert x1.shape == x2.shape, "Input images must be the same size"
        _, c, h, w = x1.shape
        x1, x2 = self.norm(x1), self.norm(x2)
        x=rearrange(x1, "b c h w -> b c (h w)")
        x=torch.var(x,dim=2,keepdim=True).squeeze(dim=2)
        score = torch.einsum("bcij,bcij->bc", x1, x2)
        score=torch.einsum("bc,bc ->b", x, score)
        score /= c * h * w
        return score

def to_patches(x, patch_size):
    x = x.unfold(2, patch_size, step=1).unfold(3, patch_size, step=1).contiguous()
    return rearrange(x, "b c p1 p2 h w -> b (c p1 p2) h w")

class NormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images."""

    def __init__(self, patch_size=None, eps=1e-5):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(num_features=1, eps=eps)
        self.patch_size = patch_size

    def forward(self, x1, x2):
        if self.patch_size is not None:
            x1 = to_patches(x1, self.patch_size)
            x2 = to_patches(x2, self.patch_size)
        assert x1.shape == x2.shape, "Input images must be the same size"
        _, c, h, w = x1.shape
        x1, x2 = self.norm(x1), self.norm(x2)
        score = torch.einsum("b...,b...->b", x1, x2)
        score /= c * h * w
        return score
    
class MultiscaleNormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images at multiple scales."""

    def __init__(self, patch_sizes=[None], patch_weights=[1.0], eps=1e-5):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(num_features=1, eps=eps)

        assert len(patch_sizes) == len(patch_weights), "Each scale must have a weight"
        self.nccs = [
            NormalizedCrossCorrelation2d(patch_size) for patch_size in patch_sizes
        ]
        self.patch_weights = patch_weights

    def forward(self, x1, x2):
        scores = []
        append=scores.append
        for weight, ncc in zip(self.patch_weights, self.nccs):
            append(weight * ncc(x1, x2))
        return torch.stack(scores, dim=0).sum(dim=0)

class GradientNormalizedCrossCorrelation2d(NormalizedCrossCorrelation2d):
    """Compute Normalized Cross Correlation between the image gradients of two batches of images."""

    def __init__(self, patch_size=None, sigma=1.0, **kwargs):
        super().__init__(patch_size, **kwargs)
        self.sobel = Sobel(sigma)

    def forward(self, x1, x2):
        return super().forward(self.sobel(x1), self.sobel(x2))


from torchvision.transforms.functional import gaussian_blur


class Sobel(torch.nn.Module):
    def __init__(self, sigma,device='cuda'):
        super().__init__()
        self.sigma = sigma
        self.filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=2,  # X- and Y-gradients
            kernel_size=3,
            stride=1,
            padding=1,  # Return images of the same size as inputs
            bias=False,
        )

        Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(torch.float32)
        Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(torch.float32)
        G = torch.stack([Gx, Gy]).unsqueeze(1)
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)
        self.filter.to(device)

    def forward(self, img):
        # x = gaussian_blur(img, 5, self.sigma)
        x = self.filter(img)
        return x
class MultiscaleGradientNormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between  the image gradients of two batches of images at multiple scales."""

    def __init__(self, patch_sizes=[None], patch_weights=[1.0], eps=1e-5):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(num_features=1, eps=eps)

        assert len(patch_sizes) == len(patch_weights), "Each scale must have a weight"
        self.nccs = [
            GradientNormalizedCrossCorrelation2d(patch_size) for patch_size in patch_sizes
        ]
        self.patch_weights = patch_weights

    def forward(self, x1, x2):
        scores = []
        append=scores.append
        for weight, ncc in zip(self.patch_weights, self.nccs):
            append(weight * ncc(x1, x2))
        return torch.stack(scores, dim=0).sum(dim=0)
if __name__=='__main__':
    image = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    # patch_size=(2,2)
    # # x = torch.randn(1, 1, 4, 4)
    # y = nn.Unfold(kernel_size=patch_size, stride=patch_size)(x)
    # print(y.size())
    # B=y.size()[0]
    # C=1
    # num_patches=y.size()[2]
    # y = y.transpose(2,1)
    

