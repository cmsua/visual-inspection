from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    PyTorch implementation of the Variance-Invariance-Covariance Regularization (VICReg) loss.
    
    Parameters
    ----------
    gamma: float
        Constant target value for the standard deviation.
    epsilon: float
        Small scalar preventing numerical instabilities.
    lambda\_: float
        Weight for the invariance loss term.
    mu: float
        Weight for the variance loss term.
    nu: float
        Weight for the covariance loss term.

    .. References::
        Adrien Bardes, Jean Ponce, and Yann LeCun.
        [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906).
        *arXiv preprint arXiv:2105.04906*, 2021.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
        lambda_: float = 25.0,
        mu: float = 25.0,
        nu: float = 1.0
    ):
        super(VICRegLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu

    def _off_diagonal(self, x: Tensor) -> Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, m + 1)[:, 1:].flatten()

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tuple[Tensor, ...]:
        batch_size, dim = z_a.shape

        # Variance loss
        std_z_a = torch.sqrt(torch.var(z_a, dim=0) + self.epsilon)
        std_z_b = torch.sqrt(torch.var(z_b, dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # Invariance loss
        sim_loss = F.mse_loss(z_a, z_b)

        # Covariance loss
        z_a = z_a - torch.mean(z_a, dim=0)
        z_b = z_b - torch.mean(z_b, dim=0)
        cov_z_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (batch_size - 1)
        cov_loss = (self._off_diagonal(cov_z_a).pow(2).sum() + self._off_diagonal(cov_z_b).pow(2).sum()) / dim

        # Total VIC loss
        loss = self.lambda_ * sim_loss + self.mu * std_loss + self.nu * cov_loss

        return loss, std_loss, sim_loss, cov_loss