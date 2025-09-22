from torch import optim


OPTIM_REGISTRY = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
    # Add more optimizers here as needed
}