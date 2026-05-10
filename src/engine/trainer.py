import csv
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import (
    is_available, is_initialized,
    get_rank, get_world_size,
    all_gather, all_gather_object
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..configs import TrainConfig
from ..loss import LOSS_REGISTRY
from ..optim import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from ..utils import (
    CALLBACK_REGISTRY,
    get_loss_from_config,
    get_optim_from_config,
    get_optim_wrapper_from_config,
    get_scheduler_from_config,
    get_callbacks_from_config,
    cleanup_ddp
)


class Trainer:
    """
    Base class for training models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        device: Optional[Union[torch.device, int]] = None,
        metric: Optional[Callable] = None,
        config: Optional[TrainConfig] = None,
        # Parameters below can override config if supplied explicitly
        batch_size: Optional[int] = None,
        criterion: Optional[Dict] = None,
        optimizer: Optional[Dict] = None,
        optimizer_wrapper: Optional[Dict] = None,
        scheduler: Optional[Dict] = None,
        callbacks: Optional[List[Dict]] = None,
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = None,
        logging_dir: Optional[str] = None,
        logging_steps: Optional[int] = None,
        eval_strategy: Optional[str] = None,
        eval_steps: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        best_model_monitor: Optional[str] = None,
        best_model_mode: Optional[str] = None,
        save_best: Optional[bool] = None,
        save_ckpt: Optional[bool] = None,
        save_fig: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None
    ):
        """
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
        device: torch.device or int, optional
            Device to run the training on. Overrides config if provided.
        metric: Callable, optional
            A function to compute a metric for evaluation.
        config: TrainConfig, optional
            Configuration object containing training parameters.
        batch_size: int, optional
            Batch size for training. Overrides config if provided.
        criterion: Dict, optional
            Loss configuration override.
        optimizer: Dict, optional
            Optimizer configuration override.
        optimizer_wrapper: Dict, optional
            Optimizer wrapper configuration override.
        scheduler: Dict, optional
            Scheduler configuration override.
        callbacks: List[Dict], optional
            Callback configuration override.
        num_epochs: int, optional
            Number of epochs override.
        start_epoch: int, optional
            Starting epoch override.
        logging_dir: str, optional
            Logging directory override.
        logging_steps: int, optional
            Training-step logging frequency override.
        eval_strategy: str, optional
            Validation cadence, either `'epoch'` or `'steps'`.
        eval_steps: int, optional
            Validation interval in optimizer steps when `eval_strategy='steps'`.
        progress_bar: bool, optional
            Whether to show tqdm progress.
        best_model_monitor: str, optional
            Validation metric key used to decide which model is written to the
            `best/` directory.
        best_model_mode: str, optional
            Whether lower (`'min'`) or higher (`'max'`) values of
            `best_model_monitor` are considered better.
        save_best: bool, optional
            Whether to save the best validation checkpoint.
        save_ckpt: bool, optional
            Whether to save checkpoints on validation events.
        save_fig: bool, optional
            Whether to save evaluation plots.
        num_workers: int, optional
            DataLoader worker count override.
        pin_memory: bool, optional
            DataLoader pin-memory override.
        """
        self.rank = 0
        self.world_size = 1

        # Prepare the device for multi-GPU training if available
        if is_available() and is_initialized():
            self.rank = get_rank()
            self.world_size = get_world_size()
            
        self.device = self._resolve_device(device=device)

        # Prepare the model for multi-GPU training if available
        self.model = model.to(self.device)
        self._is_distributed = self.world_size > 1
        if self._is_distributed:
            ddp_kwargs = {
                'module': self.model,
                'find_unused_parameters': True,
                'gradient_as_bucket_view': True
            }
            if self.device.type == 'cuda':
                ddp_kwargs['device_ids'] = [self.device.index]
                ddp_kwargs['output_device'] = self.device.index

            self.model = DDP(**ddp_kwargs)

        # Use config if provided, otherwise use defaults
        if config is not None:
            self.batch_size = batch_size if batch_size is not None else config.batch_size
            self.criterion = get_loss_from_config(criterion if criterion is not None else config.criterion, LOSS_REGISTRY)
            self.optimizer = get_optim_from_config(optimizer if optimizer is not None else config.optimizer, OPTIM_REGISTRY, self.model)
            if optimizer_wrapper is not None or config.optimizer_wrapper is not None:
                self.optimizer = get_optim_wrapper_from_config(optimizer_wrapper if optimizer_wrapper is not None else config.optimizer_wrapper, OPTIM_REGISTRY, self.optimizer)
            scheduler_config = scheduler if scheduler is not None else config.scheduler
            self.scheduler = (
                get_scheduler_from_config(scheduler_config, SCHEDULER_REGISTRY, self.optimizer)
                if scheduler_config is not None else None
            )
            callback_config = callbacks if callbacks is not None else config.callbacks
            self.callbacks = (
                get_callbacks_from_config(callback_config, CALLBACK_REGISTRY)
                if callback_config is not None else []
            )
            self.num_epochs = num_epochs if num_epochs is not None else config.num_epochs
            self.start_epoch = start_epoch if start_epoch is not None else config.start_epoch
            self.logging_dir = logging_dir if logging_dir is not None else config.logging_dir
            self.logging_steps = logging_steps if logging_steps is not None else config.logging_steps
            self.eval_strategy = eval_strategy if eval_strategy is not None else config.eval_strategy
            self.eval_steps = eval_steps if eval_steps is not None else config.eval_steps
            self.progress_bar = progress_bar if progress_bar is not None else config.progress_bar
            self.best_model_monitor = (
                best_model_monitor if best_model_monitor is not None else config.best_model_monitor
            )
            self.best_model_mode = (
                best_model_mode if best_model_mode is not None else config.best_model_mode
            )
            self.save_best = save_best if save_best is not None else config.save_best
            self.save_ckpt = save_ckpt if save_ckpt is not None else config.save_ckpt
            self.save_fig = save_fig if save_fig is not None else config.save_fig
            self.num_workers = num_workers if num_workers is not None else config.num_workers
            self.pin_memory = pin_memory if pin_memory is not None else config.pin_memory
        else:
            self.batch_size = batch_size if batch_size is not None else 64
            if criterion is None:
                raise ValueError("Criterion must be provided if config is not supplied.")
            self.criterion = get_loss_from_config(criterion, LOSS_REGISTRY)
            if optimizer is None:
                raise ValueError("Optimizer must be provided if config is not supplied.")
            self.optimizer = get_optim_from_config(optimizer, OPTIM_REGISTRY, self.model)
            if optimizer_wrapper is not None:
                self.optimizer = get_optim_wrapper_from_config(optimizer_wrapper, OPTIM_REGISTRY, self.optimizer)
            self.scheduler = get_scheduler_from_config(scheduler, SCHEDULER_REGISTRY, self.optimizer) if scheduler else None
            self.callbacks = get_callbacks_from_config(callbacks, CALLBACK_REGISTRY) if callbacks is not None else None
            self.num_epochs = num_epochs if num_epochs is not None else 20
            self.start_epoch = start_epoch if start_epoch is not None else 0
            self.logging_dir = logging_dir if logging_dir is not None else 'logs'
            self.logging_steps = logging_steps if logging_steps is not None else 25
            self.eval_strategy = eval_strategy if eval_strategy is not None else 'epoch'
            self.eval_steps = eval_steps if eval_steps is not None else 0
            self.progress_bar = progress_bar if progress_bar is not None else True
            self.best_model_monitor = best_model_monitor if best_model_monitor is not None else 'val_loss'
            self.best_model_mode = best_model_mode if best_model_mode is not None else 'min'
            self.save_best = save_best if save_best is not None else True
            self.save_ckpt = save_ckpt if save_ckpt is not None else True
            self.save_fig = save_fig if save_fig is not None else False
            self.num_workers = num_workers if num_workers is not None else 0
            self.pin_memory = pin_memory if pin_memory is not None else False

        self.callbacks = self.callbacks if self.callbacks is not None else []
        self._validate_eval_config()
        self._validate_best_model_config()
        
        # Initialize data samplers
        train_sampler = self._build_sampler(train_dataset, shuffle=True)
        val_sampler = self._build_sampler(val_dataset, shuffle=False)
        test_sampler = self._build_sampler(test_dataset, shuffle=False) if test_dataset is not None else None

        # Initialize data loaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=(not self._is_distributed),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) if test_dataset is not None else None

        # Initialize metrics and history
        self.metric = metric
        self.history = {
            'eval_strategy': self.eval_strategy,
            'epoch': [],
            'step': [],
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': []
        }
        self.best_model_score = self._best_model_initial_score()
        self._resume_global_step = 0
        self._resume_batch_offset = 0
        self._resume_epoch_state = None

        # Initialize the logging directory
        self.model_name = self.model_module.__class__.__name__
            
        os.makedirs(self.logging_dir, exist_ok=True)
        self.log_dir = os.path.join(self.logging_dir, self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Subfolders
        self.best_models_dir = os.path.join(self.log_dir, 'best')
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.loggings_dir = os.path.join(self.log_dir, 'logging')
        self.outputs_dir = os.path.join(self.log_dir, 'output')
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.loggings_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Determine run index
        self.run_name = self._get_next_run_index()

        # Logging and best model paths
        self._log_header_written = False
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def _get_next_run_index(self) -> str:
        return f"pid{os.getpid()}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    @property
    def model_module(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _resolve_device(self, device: Optional[Union[torch.device, int]]) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device(f'cuda:{self.rank}')

            return torch.device('cpu')

        if isinstance(device, int):
            return torch.device(f'cuda:{device}')

        return device

    def _build_sampler(self, dataset: Optional[Dataset], shuffle: bool) -> Optional[DistributedSampler]:
        if dataset is None or not self._is_distributed:
            return None

        return DistributedSampler(dataset, shuffle=shuffle)

    def _validate_eval_config(self) -> None:
        if self.eval_strategy not in {'epoch', 'steps'}:
            raise ValueError("eval_strategy must be either 'epoch' or 'steps'.")
        if self.eval_strategy == 'steps' and int(self.eval_steps) <= 0:
            raise ValueError("eval_steps must be greater than 0 when eval_strategy='steps'.")

    def _validate_best_model_config(self) -> None:
        if self.best_model_mode not in {'min', 'max'}:
            raise ValueError("best_model_mode must be either 'min' or 'max'.")
        if not isinstance(self.best_model_monitor, str) or not self.best_model_monitor:
            raise ValueError("best_model_monitor must be a non-empty string.")

    def _best_model_initial_score(self) -> float:
        if self.best_model_mode == 'min':
            return float('inf')

        return float('-inf')

    def _is_better_best_model_score(self, score: float) -> bool:
        if self.best_model_mode == 'min':
            return score < self.best_model_score

        return score > self.best_model_score

    def _extract_best_model_score(self, logs: Dict[str, float]) -> float:
        if self.best_model_monitor not in logs:
            raise KeyError(
                f"best_model_monitor '{self.best_model_monitor}' was not found in validation logs."
            )

        return float(logs[self.best_model_monitor])

    def _set_logging_paths(self, run_name: str):
        self.run_name = run_name
        self._log_header_written = True
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def save_checkpoint(
        self,
        epoch: int,
        global_step: Optional[int] = None,
        epoch_progress: Optional[float] = None,
        epoch_state: Optional[Dict[str, float]] = None
    ):
        model_state = self.model_module.state_dict()
        if self.checkpoint_path and self.rank == 0:
            checkpoint = {
                'run_name': self.run_name,
                'epoch': epoch,
                'global_step': int(global_step) if global_step is not None else None,
                'epoch_progress': float(epoch_progress) if epoch_progress is not None else None,
                'epoch_state': dict(epoch_state) if epoch_state is not None else None,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._set_logging_paths(checkpoint['run_name'])
        self.model_module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint['history']
        if self.best_model_monitor in self.history and self.history[self.best_model_monitor]:
            history_scores = [float(score) for score in self.history[self.best_model_monitor]]
            if self.best_model_mode == 'min':
                self.best_model_score = min(history_scores)
            else:
                self.best_model_score = max(history_scores)
        else:
            self.best_model_score = self._best_model_initial_score()

        checkpoint_global_step = checkpoint.get('global_step')
        if checkpoint_global_step is not None:
            steps_per_epoch = len(self.train_loader)
            self._resume_global_step = int(checkpoint_global_step)
            self.start_epoch = int(self._resume_global_step // max(steps_per_epoch, 1))
            self._resume_batch_offset = int(self._resume_global_step % max(steps_per_epoch, 1))
            epoch_state = checkpoint.get('epoch_state')
            self._resume_epoch_state = dict(epoch_state) if epoch_state is not None else None
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self._resume_global_step = int(self.start_epoch * len(self.train_loader))
            self._resume_batch_offset = 0
            self._resume_epoch_state = None
    
    def load_best_model(self, best_model_path: str):
        run_name = os.path.splitext(os.path.basename(best_model_path))[0]
        self._set_logging_paths(run_name)
        self.model_module.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def log_csv(self, log_dict: Dict[str, float]):
        if self.rank != 0:
            return
        
        write_header = not self._log_header_written
        with open(self.logging_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
            if write_header:
                writer.writeheader()
                self._log_header_written = True

            writer.writerow(log_dict)

    def _init_epoch_state(self) -> Dict[str, float]:
        return {
            'loss_sum': 0.0,
            'metric_sum': 0.0,
            'count': 0
        }

    def _update_epoch_state(
        self,
        epoch_state: Dict[str, float],
        batch_size: int,
        loss: torch.Tensor,
        metric_value: float
    ) -> None:
        epoch_state['loss_sum'] += float(loss.item()) * batch_size
        epoch_state['metric_sum'] += float(metric_value) * batch_size
        epoch_state['count'] += int(batch_size)

    def _reduce_epoch_state(self, epoch_state: Dict[str, float]) -> Dict[str, float]:
        local_tensor = torch.tensor(
            [
                epoch_state['loss_sum'],
                epoch_state['metric_sum'],
                float(epoch_state['count'])
            ],
            dtype=torch.float64,
            device=self.device
        )
        if self._is_distributed:
            gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
            all_gather(gathered, local_tensor)
            reduced_tensor = torch.stack(gathered, dim=0).sum(dim=0)
        else:
            reduced_tensor = local_tensor

        count = max(int(reduced_tensor[2].item()), 1)
        return {
            'loss': float(reduced_tensor[0].item()) / count,
            'metric': float(reduced_tensor[1].item()) / count
        }

    def _run_validation_loop(self) -> Dict[str, float]:
        self.model.eval()
        epoch_state = self._init_epoch_state()

        with torch.no_grad():
            for X_val, y_val in self.val_loader:
                X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                y_val = y_val.to(self.device, non_blocking=self.pin_memory)

                outputs_val = self.model(X_val)
                loss_val = self.criterion(outputs_val, y_val)
                batch_metric = float(self.metric(outputs_val, y_val)) if self.metric else 0.0
                self._update_epoch_state(
                    epoch_state=epoch_state,
                    batch_size=int(y_val.size(0)),
                    loss=loss_val,
                    metric_value=batch_metric
                )

        return self._reduce_epoch_state(epoch_state)

    def _update_history(
        self,
        epoch_progress: float,
        global_step: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        self.history['epoch'].append(epoch_progress)
        self.history['step'].append(global_step)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_metric'].append(train_metrics['metric'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_metric'].append(val_metrics['metric'])

    def _step_scheduler(self, val_metrics: Dict[str, float]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_metrics['loss'])
            return

        self.scheduler.step()

    def _finalize_validation_event(
        self,
        epoch_index: int,
        epoch_progress: float,
        global_step: int,
        epoch_state: Dict[str, float],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        if self.rank == 0:
            tqdm.write(
                f"epoch: {epoch_progress:.4f}/{self.num_epochs} | "
                f"val_loss: {val_metrics['loss']:.4f} | "
                f"val_metric: {val_metrics['metric']:.4f}"
            )

        logs = {
            'epoch': epoch_progress,
            'step': global_step,
            'train_loss': train_metrics['loss'],
            'train_metric': train_metrics['metric'],
            'val_loss': val_metrics['loss'],
            'val_metric': val_metrics['metric'],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

        monitored_score = self._extract_best_model_score(logs)
        if self.best_model_path and self.rank == 0 and self._is_better_best_model_score(monitored_score):
            self.best_model_score = monitored_score
            torch.save(self.model_module.state_dict(), self.best_model_path)

        self._step_scheduler(val_metrics)
        logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
        self._update_history(
            epoch_progress=epoch_progress,
            global_step=global_step,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )
        self.log_csv(logs)
        if self.save_ckpt:
            self.save_checkpoint(
                epoch=epoch_index,
                global_step=global_step,
                epoch_progress=epoch_progress,
                epoch_state=epoch_state
            )

        return logs

    def _run_validation_callbacks(self, epoch: int, logs: Dict[str, float]) -> bool:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, trainer=self, logs=logs)

        return any(getattr(callback, 'should_stop', False) for callback in self.callbacks)

    def _should_run_step_validation(self, step: int, total_steps: int) -> bool:
        if self.eval_strategy != 'steps':
            return False
        if step == total_steps:
            return True

        return step % int(self.eval_steps) == 0

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        epoch = self.start_epoch
        try:
            # Callback before training
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self._resume_global_step
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
                if epoch == self.start_epoch and self._resume_batch_offset > 0 and self._resume_epoch_state is not None:
                    epoch_state = dict(self._resume_epoch_state)
                else:
                    epoch_state = self._init_epoch_state()
                last_eval_step = None
                should_stop = False
                resume_batch_offset = self._resume_batch_offset if epoch == self.start_epoch else 0

                for batch_idx, (X, y) in enumerate(self.train_loader):
                    if batch_idx < resume_batch_offset:
                        continue

                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    self.optimizer.zero_grad()
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                    batch_metric = float(self.metric(outputs, y)) if self.metric else 0.0
                    self._update_epoch_state(
                        epoch_state=epoch_state,
                        batch_size=int(y.size(0)),
                        loss=loss,
                        metric_value=batch_metric
                    )
                    reduced_state = self._reduce_epoch_state(epoch_state)
                    
                    # Short summary for training
                    if self.rank == 0:
                        if step % self.logging_steps == 0 or step == total_steps:
                            tqdm.write(
                                f"step: {step}/{total_steps} | "
                                f"train_loss: {reduced_state['loss']:.4f} | "
                                f"train_metric: {reduced_state['metric']:.4f}"
                            )

                        global_bar.set_postfix({
                            'epoch': f'{epoch + 1}/{self.num_epochs}',
                            'avg_loss': f"{reduced_state['loss']:.4f}",
                            'avg_metric': f"{reduced_state['metric']:.4f}"
                        })

                    global_bar.update(1)

                    if self._should_run_step_validation(step=step, total_steps=total_steps):
                        train_metrics = self._reduce_epoch_state(epoch_state)
                        val_metrics = self._run_validation_loop()
                        epoch_progress = epoch + ((batch_idx + 1) / max(len(self.train_loader), 1))
                        logs = self._finalize_validation_event(
                            epoch_index=epoch,
                            epoch_progress=epoch_progress,
                            global_step=step,
                            epoch_state=epoch_state,
                            train_metrics=train_metrics,
                            val_metrics=val_metrics
                        )
                        last_eval_step = step
                        should_stop = self._run_validation_callbacks(epoch=epoch, logs=logs)
                        if should_stop:
                            break

                if should_stop:
                    break

                self._resume_batch_offset = 0
                self._resume_epoch_state = None
                train_metrics = self._reduce_epoch_state(epoch_state)
                if self.eval_strategy == 'epoch':
                    val_metrics = self._run_validation_loop()
                    logs = self._finalize_validation_event(
                        epoch_index=epoch,
                        epoch_progress=epoch + 1,
                        global_step=(epoch + 1) * len(self.train_loader),
                        epoch_state=epoch_state,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics
                    )
                    should_stop = self._run_validation_callbacks(epoch=epoch, logs=logs)

                if should_stop:
                    break

            # Callback after training
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)
        except KeyboardInterrupt:
            if self.rank == 0:
                print(f"\nTraining interrupted at epoch {epoch + 1}.")
                
            cleanup_ddp()

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
        loss_sum = 0.0
        metric_sum = 0.0
        count = 0
        local_true = []
        local_pred = []

        for X_test, y_test in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)
            y_test = y_test.to(self.device, non_blocking=self.pin_memory)

            outputs = self.model(X_test)
            batch_loss = self.criterion(outputs, y_test)
            bsz = y_test.size(0)
            loss_sum += float(batch_loss.item()) * bsz

            if self.metric:
                metric_sum += float(self.metric(outputs, y_test)) * bsz

            count += bsz

            probs = outputs
            if loss_type == 'cross_entropy':
                probs = F.softmax(outputs, dim=1)
            elif loss_type == 'bce':
                probs = torch.sigmoid(outputs)

            local_true.append(y_test.detach().cpu().numpy())
            local_pred.append(probs.detach().cpu().numpy())

        # Stack per-rank arrays
        local_true = np.concatenate(local_true, axis=0) if local_true else np.empty((0,))
        local_pred = np.concatenate(local_pred, axis=0) if local_pred else np.empty((0,))

        # Gather scalars from all processes
        if self._is_distributed:
            packed = torch.tensor(
                data=[loss_sum, metric_sum, float(count)],
                dtype=torch.float64,
                device=self.device,
            )
            gathered = [torch.zeros_like(packed) for _ in range(self.world_size)]
            all_gather(gathered, packed)

            total_loss_sum = sum(g[0].item() for g in gathered)
            total_metric_sum = sum(g[1].item() for g in gathered)
            total_count = int(sum(g[2].item() for g in gathered))
        else:
            total_loss_sum = loss_sum
            total_metric_sum = metric_sum
            total_count = count

        # Global averages
        test_loss = total_loss_sum / max(total_count, 1)
        test_metric = (total_metric_sum / max(total_count, 1)) if self.metric else 0.0

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

        self.last_eval_metrics = {
            'loss': test_loss,
            'metric': test_metric
        }

        if self.rank == 0:
            print(f"test_loss: {test_loss:.4f} | test_metric: {test_metric:.4f}")

            # Visualization
            if plot is not None:
                if isinstance(plot, list):
                    for i, viz in enumerate(plot):
                        output_path = os.path.join(self.outputs_dir, f"{self.run_name}_viz_{i + 1}.png")
                        output_path = output_path if self.save_fig else None
                        viz(y_true, y_pred, save_fig=output_path)
                else:
                    output_path = os.path.join(self.outputs_dir, f"{self.run_name}.png")
                    output_path = output_path if self.save_fig else None
                    plot(y_true, y_pred, save_fig=output_path)

        return test_loss, test_metric, y_true, y_pred
