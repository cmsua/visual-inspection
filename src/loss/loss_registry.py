from torch import nn


LOSS_REGISTRY = {
    'cross_entropy_loss': nn.CrossEntropyLoss,
    'bce_with_logits_loss': nn.BCEWithLogitsLoss,
    'mse_loss': nn.MSELoss,
    # Add more loss functions here as needed
}