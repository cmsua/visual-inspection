from typing import List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    height: int
    width: int
    latent_dim: int
    kernel_sizes: List[int]