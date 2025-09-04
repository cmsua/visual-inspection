import os
import re
import csv
from typing import List, Tuple, Dict, Callable, Optional, Union

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader

from ..configs import TrainConfig
from ..utils import BaseCallback


class Trainer:
    """
    Base class for training models.
    
    Parameters
    ----------
    model: nn.Module
        The model to be trained.
    train_dataset: Dataset
        The dataset for training.
    val_dataset: Dataset
        The dataset for validation.
    test_dataset: Dataset, optional
        The dataset for testing.
    criterion: _Loss
        The loss function to be used.
    optimizer: Optimizer
        The optimizer for training.
    scheduler: _LRScheduler, optional
        The learning rate scheduler.
    metric: Callable, optional
        The metric function to evaluate the model (e.g. accuracy).
    callbacks: List[BaseCallback], optional
        List of callbacks to be executed during training.
    config: TrainConfig, optional
        Configuration object containing training parameters.
    batch_size: int, optional
        Batch size for training and validation.
    num_epochs: int, optional
        Number of training epochs.
    start_epoch: int, optional
        The epoch to start training from.
    history: Dict[str, List[float]], optional
        History of training metrics.
    logging_dir: str, optional
        Directory for logging training progress.
    logging_steps: int, optional
        Frequency of logging training progress.
    save_best: bool, optional
        Whether to save the best model based on validation loss.
    save_ckpt: bool, optional
        Whether to save checkpoints during training.
    device: torch.device, optional
        Device to run the training on (CPU or GPU).
    num_workers: int, optional
        Number of workers for data loading.
    pin_memory: bool, optional
        Whether to pin memory for data loading.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset],
        criterion: _Loss,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        metric: Optional[Callable] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        config: Optional[TrainConfig] = None,
        # Parameters below can override config if supplied explicitly
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = None,
        history: Optional[Dict[str, List[float]]] = None,
        logging_dir: Optional[str] = None,
        logging_steps: Optional[int] = None,
        save_best: Optional[bool] = None,
        save_ckpt: Optional[bool] = None,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.callbacks = callbacks or []

        # Use config if provided, otherwise use defaults
        if config is not None:
            self.batch_size = batch_size if batch_size is not None else config.batch_size
            self.num_epochs = num_epochs if num_epochs is not None else config.num_epochs
            self.start_epoch = start_epoch if start_epoch is not None else config.start_epoch
            self.logging_dir = logging_dir if logging_dir is not None else config.logging_dir
            self.logging_steps = logging_steps if logging_steps is not None else config.logging_steps
            self.save_best = save_best if save_best is not None else config.save_best
            self.save_ckpt = save_ckpt if save_ckpt is not None else config.save_ckpt
            self.device = device if device is not None else torch.device(config.device)
            self.num_workers = num_workers if num_workers is not None else config.num_workers
            self.pin_memory = pin_memory if pin_memory is not None else config.pin_memory
        else:
            self.batch_size = batch_size if batch_size is not None else 64
            self.num_epochs = num_epochs if num_epochs is not None else 20
            self.start_epoch = start_epoch if start_epoch is not None else 0
            self.logging_dir = logging_dir if logging_dir is not None else 'logs'
            self.logging_steps = logging_steps if logging_steps is not None else 500
            self.save_best = save_best if save_best is not None else True
            self.save_ckpt = save_ckpt if save_ckpt is not None else True
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.num_workers = num_workers if num_workers is not None else 0
            self.pin_memory = pin_memory if pin_memory is not None else False

        # Initialize data loaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) if test_dataset is not None else None
        self.history = history or {
            'epoch': [],
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': []
        }

        self.best_val_loss = min(self.history['val_loss']) if self.history['val_loss'] else float('inf')

        self.model_name = self.model.__class__.__name__
        os.makedirs(self.logging_dir, exist_ok=True)
        self.log_dir = os.path.join(self.logging_dir, self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Subfolders
        self.best_models_dir = os.path.join(self.log_dir, 'best')
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.loggings_dir = os.path.join(self.log_dir, 'loggings')
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.loggings_dir, exist_ok=True)

        # Determine run index
        run_index = self._get_next_run_index(self.loggings_dir, 'run', '.csv')
        self.run_name = f"run_{run_index:02d}"

        # Logging and best model paths
        self._log_header_written = False
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def _get_next_run_index(self, directory: str, prefix: str, suffix: str) -> int:
        os.makedirs(directory, exist_ok=True)
        existing = [
            f for f in os.listdir(directory)
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        indices = [
            int(m.group(1)) for f in existing
            if (m := re.search(rf"{prefix}_(\d+)", f))
        ]
        return max(indices, default=0) + 1

    def save_checkpoint(self, epoch: int):
        if self.checkpoint_path:
            checkpoint = {
                'run_name': self.run_name,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.run_name = checkpoint['run_name']
        self._log_header_written = True
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch']
        self.history = checkpoint['history']

        return self.start_epoch
    
    def load_best_model(self, best_model_path: str):
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def log_csv(self, log_dict: Dict[str, float]):
        write_header = not self._log_header_written
        with open(self.logging_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
            if write_header:
                writer.writeheader()
                self._log_header_written = True
            writer.writerow(log_dict)

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            # Callback before training
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)
            global_bar = tqdm(
                total=total_steps,
                initial=start_step,
                desc="Training",
                dynamic_ncols=True
            )

            for epoch in range(self.start_epoch, self.num_epochs):
                # Callback at the beginning of each epoch
                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch, trainer=self)

                # Training phase
                self.model.train()
                running_loss = 0.0
                running_metric = 0.0

                for batch_idx, (X, y) in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    self.optimizer.zero_grad()

                    outputs = self.model(X)

                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    if self.metric:
                        running_metric += self.metric(outputs, y)

                    avg_loss = running_loss / (batch_idx + 1)
                    avg_metric = running_metric / (batch_idx + 1)

                    # Short summary
                    if step % self.logging_steps == 0 or step == total_steps:
                        tqdm.write(
                            f"step: {step}/{total_steps} | "
                            f"train_loss: {avg_loss:.4f} | "
                            f"train_metric: {avg_metric:.4f}"
                        )

                    global_bar.set_postfix({
                        "epoch": f"{epoch + 1}/{self.num_epochs}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "avg_metric": f"{avg_metric:.4f}"
                    })
                    global_bar.update(1)

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_metric = 0.0

                with torch.no_grad():
                    for X_val, y_val in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        y_val = y_val.to(self.device, non_blocking=self.pin_memory)

                        outputs_val = self.model(X_val)

                        val_loss += self.criterion(outputs_val, y_val).item()

                        if self.metric:
                            val_metric += self.metric(outputs_val, y_val)

                val_loss /= len(self.val_loader)
                val_metric /= len(self.val_loader)

                # Short summary for validation
                tqdm.write(
                    f"epoch: {epoch + 1}/{self.num_epochs} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"val_metric: {val_metric:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()

                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)

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
        loss_type: str,
        plot: Optional[Union[Callable, List[Callable]]] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")
        
        self.model.eval()
        test_loss = 0.0
        test_metric = 0.0
        y_true_list = []
        y_pred_list = []

        for X_test, y_test in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)
            y_test = y_test.to(self.device, non_blocking=self.pin_memory)

            outputs_test = self.model(X_test)

            test_loss += self.criterion(outputs_test, y_test).item()

            if self.metric:
                test_metric += self.metric(outputs_test, y_test)

            probs = outputs_test
            if loss_type == 'cross_entropy':
                probs = F.softmax(outputs_test, dim=1)
            elif loss_type == 'bce':
                probs = torch.sigmoid(outputs_test)

            y_true_list.append(y_test.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())

        test_loss /= len(self.test_loader)
        test_metric /= len(self.test_loader)

        print(f"test_loss: {test_loss:.4f} | test_metric: {test_metric:.4f}")

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        # Visualization
        if plot is not None:
            if isinstance(plot, list):
                for viz in plot:
                    viz(y_true, y_pred)
            else:
                plot(y_true, y_pred)

        return test_loss, test_metric, y_true, y_pred
    

class AutoencoderTrainer(Trainer):
    """
    Trainer specifically for autoencoder models.
    Inherits from Trainer and implements the train method for autoencoders.
    
    Parameters
    ----------
    model: nn.Module
        The model to be trained.
    train_dataset: Dataset
        The dataset for training.
    val_dataset: Dataset
        The dataset for validation.
    test_dataset: Dataset
        The dataset for testing.
    criterion: _Loss
        The loss function to be used.
    optimizer: Optimizer
        The optimizer for training.
    scheduler: _LRScheduler, optional
        The learning rate scheduler.
    metric: Callable, optional
        The metric function to evaluate the model (e.g. accuracy).
    callbacks: List[BaseCallback], optional
        List of callbacks to be executed during training.
    config: TrainConfig, optional
        Configuration object containing training parameters.
    batch_size: int, optional
        Batch size for training and validation.
    num_epochs: int, optional
        Number of training epochs.
    start_epoch: int, optional
        The epoch to start training from.
    history: Dict[str, List[float]], optional
        History of training metrics.
    logging_dir: str, optional
        Directory for logging training progress.
    logging_steps: int, optional
        Frequency of logging training progress.
    save_best: bool, optional
        Whether to save the best model based on validation loss.
    save_ckpt: bool, optional
        Whether to save checkpoints during training.
    device: torch.device, optional
        Device to run the training on (CPU or GPU).
    num_workers: int, optional
        Number of workers for data loading.
    pin_memory: bool, optional
        Whether to pin memory for data loading.
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
            global_bar = tqdm(
                total=total_steps,
                initial=start_step,
                desc="Training",
                dynamic_ncols=True
            )

            for epoch in range(self.start_epoch, self.num_epochs):
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
                val_loss = 0.0
                val_metric = 0.0

                with torch.no_grad():
                    for X_val in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)

                        outputs_val = self.model(X_val)

                        val_loss += self.criterion(outputs_val, X_val).item()

                        if self.metric:
                            val_metric += self.metric(outputs_val, X_val)

                val_loss /= len(self.val_loader)
                val_metric /= len(self.val_loader)

                # Short summary for validation
                tqdm.write(
                    f"epoch: {epoch + 1}/{self.num_epochs} | "
                    f"val_loss: {val_loss:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()

                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)

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
        test_loss = 0.0
        test_metric = 0.0
        y_true_list = []
        y_pred_list = []

        for X_test in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)

            outputs_test = self.model(X_test)

            test_loss += self.criterion(outputs_test, X_test).item()

            y_true_list.append(X_test.cpu().numpy())
            y_pred_list.append(torch.sigmoid(outputs_test).cpu().numpy())

        test_loss /= len(self.test_loader)

        print(f"test_loss: {test_loss:.4f}")

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        # Visualization
        if plot is not None:
            if isinstance(plot, list):
                for viz in plot:
                    viz(y_true, y_pred)
            else:
                plot(y_true, y_pred)

        return test_loss, y_true, y_pred