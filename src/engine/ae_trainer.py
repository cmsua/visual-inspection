from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from torch import Tensor

from .trainer import Trainer


class AutoencoderTrainer(Trainer):
    """
    Train reconstruction models using the base trainer's execution loop.

    Tensor batches serve as both input and target. Scheduler timing, validation,
    checkpointing, distributed reductions, callbacks, and graceful interruption
    are inherited from `Trainer`.

    Parameters
    ----------
    model : nn.Module
        Reconstruction model returning logits with the input spatial shape.
    train_dataset : Dataset
        Dataset yielding normalized RGB segment tensors.
    val_dataset : Dataset
        Validation dataset with the same contract.
    config : TrainConfig, optional
        Shared trainer settings; use a reconstruction criterion.
    """

    def _prepare_batch(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            raise TypeError("Autoencoder batches must be tensors used as both input and target.")

        tensor = self._move_batch_to_device(batch)
        return tensor, tensor

    def evaluate(self, plot: Optional[Union[Callable, List[Callable]]] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate reconstructions and retain the existing three-value API.

        Parameters
        ----------
        plot : Callable or List[Callable], optional
            Plot functions accepting targets, predictions, and `savefig`.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            Mean loss, target images, and sigmoid reconstructions.
        """
        loss, _, targets, predictions = super().evaluate(loss_type='bce', plot=plot)
        return loss, targets, predictions
