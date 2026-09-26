from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AutoencoderConfig:
    """
    CNN reconstruction geometry and architecture.

    Parameters
    ----------
    height : int
        Segment height in pixels.
    width : int
        Segment width in pixels.
    """

    height: int
    width: int
    latent_dim: int
    init_filters: int
    layers: List[int]

    @classmethod
    def from_dict(cls, d: Dict) -> 'AutoencoderConfig':
        return cls(**d)
