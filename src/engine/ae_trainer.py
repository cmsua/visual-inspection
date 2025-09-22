import os
from typing import List, Tuple, Dict, Callable, Optional, Union

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.distributed import all_gather, all_gather_object
from torch.utils.data.distributed import DistributedSampler

from .trainer import Trainer


class AutoencoderTrainer(Trainer):
    """
    Trainer specifically for autoencoder models.
    Inherits from Trainer and implements the train method for autoencoders.
    
    Parameters
    ----------
    model: nn.Module
        The model to train.
    train_dataset: Dataset
        The dataset to use for training.
    val_dataset: Dataset
        The dataset to use for validation.
    test_dataset: Dataset, optional
        The dataset to use for testing.
    device: torch.device, optional
        Device to run the training on. Overrides config if provided.
    metric: Callable, optional
        A function to compute a metric for evaluation.
    config: TrainConfig, optional
        Configuration object containing training parameters.
    batch_size: int, optional
        Batch size for training. Overrides config if provided.
    criterion: Dict, optional
        Loss function configuration. Overrides config if provided.
    optimizer: Dict, optional
        Optimizer configuration. Overrides config if provided.
    scheduler: Dict, optional
        Learning rate scheduler configuration. Overrides config if provided.
    callbacks: List[Dict], optional
        A list of callbacks to execute during training. Overrides config if provided.
    num_epochs: int, optional
        Number of epochs to train for. Overrides config if provided.
    start_epoch: int, optional
        Epoch to start training from. Overrides config if provided.
    history: Dict[str, List[float]], optional
        History of training metrics. If not provided, initializes an empty history.
    logging_dir: str, optional
        Directory to save logs. Overrides config if provided.
    logging_steps: int, optional
        Frequency of logging during training. Overrides config if provided.
    progress_bar: bool, optional
        Whether to display a tqdm progress bar. Useful to disable on HPC.
    save_best: bool, optional
        Whether to save the best model based on validation loss. Overrides config if provided.
    save_ckpt: bool, optional
        Whether to save checkpoints during training. Overrides config if provided.
    save_fig: bool, optional
        Whether to save evaluation figures. Overrides config if provided.
    num_workers: int, optional
        Number of workers for data loading. Overrides config if provided.
    pin_memory: bool, optional
        Whether to use pinned memory for data loading. Overrides config if provided.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            # Callback before training
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)
            if self.progress_bar and self.rank == 0:
                global_bar = tqdm(
                    total=total_steps,
                    initial=start_step,
                    desc="Training",
                    dynamic_ncols=True
                )
            else:
                class _NoOpBar:
                    def set_postfix(self, *args, **kwargs): pass
                    def update(self, *args, **kwargs): pass
                global_bar = _NoOpBar()

            for epoch in range(self.start_epoch, self.num_epochs):
                # Make DistributedSampler shuffle with a different seed each epoch
                if self._is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

                # Callback at the beginning of each epoch
                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch, trainer=self)

                # Training phase
                self.model.train()
                running_loss = 0.0
                running_metric = 0.0

                for batch_idx, X in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)

                    self.optimizer.zero_grad()
                    outputs = self.model(X)
                    loss = self.criterion(outputs, X)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    if self.metric:
                        running_metric += self.metric(outputs, X)

                    avg_loss = running_loss / (batch_idx + 1)
                    avg_metric = running_metric / (batch_idx + 1)

                    # Short summary
                    if self.rank == 0: 
                        if step % self.logging_steps == 0 or step == total_steps:
                            tqdm.write(
                                f"step: {step}/{total_steps} | "
                                f"train_loss: {avg_loss:.4f}"
                            )

                        global_bar.set_postfix({
                            "epoch": f"{epoch + 1}/{self.num_epochs}",
                            "avg_loss": f"{avg_loss:.4f}"
                        })

                    global_bar.update(1)

                # Validation phase
                self.model.eval()
                val_loss_sum = 0.0
                val_metric_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for X_val in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)

                        outputs_val = self.model(X_val)
                        loss_val = self.criterion(outputs_val, X_val)
                        bsz = X_val.size(0)
                        val_loss_sum += float(loss_val.item()) * bsz

                        if self.metric:
                            val_metric_sum += self.metric(outputs_val, X_val)

                        val_count += bsz
                
                # Gather validation results from all processes
                if self._is_distributed:
                    packed = torch.tensor(
                        data=[val_loss_sum, val_metric_sum, float(val_count)],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    gathered = [torch.zeros_like(packed) for _ in range(self.world_size)]
                    all_gather(gathered, packed)

                    total_loss_sum = sum(g[0].item() for g in gathered)
                    total_metric_sum = sum(g[1].item() for g in gathered)
                    total_count = int(sum(g[2].item() for g in gathered))
                else:
                    total_loss_sum = val_loss_sum
                    total_metric_sum = val_metric_sum
                    total_count = val_count

                # Global averages
                val_loss = total_loss_sum / max(total_count, 1)
                val_metric = (total_metric_sum / max(total_count, 1)) if self.metric else 0.0

                # Short summary for validation
                if self.rank == 0:
                    tqdm.write(
                        f"epoch: {epoch + 1}/{self.num_epochs} | "
                        f"val_loss: {val_loss:.4f}"
                    )

                if self.scheduler:
                    self.scheduler.step()

                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and self.rank == 0 and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    to_save = self.model.module if self._is_distributed else self.model
                    torch.save(to_save.state_dict(), self.best_model_path)

                # Update history
                self.history['epoch'].append(epoch + 1)
                self.history['train_loss'].append(avg_loss)
                self.history['train_metric'].append(avg_metric)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                # Save a checkpoint every epoch
                self.save_checkpoint(epoch)

                # Log results
                logs = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_metric': avg_metric,
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                    'learning_rate': current_lr,
                }
                self.log_csv(logs)

                # Callback after each epoch
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, trainer=self, logs=logs)

                # Break if any callback says to stop
                if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                    break

            # Callback after training
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {epoch + 1}. Saving current checkpoint.")
            self.save_checkpoint(epoch)

        return self.history, self.model
    
    @torch.no_grad()
    def evaluate(
        self,
        plot: Optional[Union[Callable, List[Callable]]] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")
        
        self.model.eval()
        loss_sum = 0.0
        count = 0
        local_true = []
        local_pred = []

        for X_test in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)

            outputs = self.model(X_test)
            batch_loss = self.criterion(outputs, X_test)
            bsz = X_test.size(0)
            loss_sum += float(batch_loss.item()) * bsz
            count += bsz

            local_true.append(X_test.detach().cpu().numpy())
            local_pred.append(torch.sigmoid(outputs).detach().cpu().numpy())

        # Stack per-rank arrays
        local_true = np.concatenate(local_true, axis=0) if local_true else np.empty((0,))
        local_pred = np.concatenate(local_pred, axis=0) if local_pred else np.empty((0,))

        # Gather scalars from all processes
        if self._is_distributed:
            packed = torch.tensor(
                data=[loss_sum, float(count)],
                dtype=torch.float64,
                device=self.device,
            )
            gathered = [torch.zeros_like(packed) for _ in range(self.world_size)]
            all_gather(gathered, packed)

            total_loss_sum = sum(g[0].item() for g in gathered)
            total_count = int(sum(g[1].item() for g in gathered))
        else:
            total_loss_sum = loss_sum
            total_count = count

        # Global average
        test_loss = total_loss_sum / max(total_count, 1)

        # Gather variable-length arrays from all processes
        if self._is_distributed:
            gathered_true = [None for _ in range(self.world_size)]
            gathered_pred = [None for _ in range(self.world_size)]
            all_gather_object(gathered_true, local_true)
            all_gather_object(gathered_pred, local_pred)

            if self.rank == 0:
                y_true = np.concatenate(gathered_true, axis=0) if gathered_true else np.empty((0,))
                y_pred = np.concatenate(gathered_pred, axis=0) if gathered_pred else np.empty((0,))
            else:
                y_true = local_true
                y_pred = local_pred
        else:
            y_true = local_true
            y_pred = local_pred

        if self.rank == 0:
            print(f"test_loss: {test_loss:.4f}")

            # Visualization
            if plot is not None:
                if isinstance(plot, list):
                    for i, viz in enumerate(plot):
                        output_path = os.path.join(self.outputs_dir, f"{self.run_name}_viz_{i + 1}.png")
                        viz(y_true, y_pred, save_fig=output_path)
                else:
                    output_path = os.path.join(self.outputs_dir, f"{self.run_name}.png")
                    plot(y_true, y_pred, save_fig=output_path)

        return test_loss, y_true, y_pred