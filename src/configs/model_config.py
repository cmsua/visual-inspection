from typing import List, Dict
from dataclasses import dataclass


@dataclass
class AutoencoderConfig:
    height: int
    width: int
    latent_dim: int
    init_filters: int
    layers: List[int]

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)