from torch import optim

from .lars import LARS
from .lookahead import Lookahead


OPTIM_REGISTRY = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'lars': LARS,
    'lookahead': Lookahead,
    'radam': optim.RAdam,
    'sgd': optim.SGD,
    # Add more optimizers here as needed
}