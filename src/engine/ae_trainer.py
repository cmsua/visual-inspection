import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import Tensor, nn
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .trainer import Trainer
from ..utils import cleanup_ddp


class AutoencoderTrainer(Trainer):
    """
    Trainer specifically for autoencoder models.
    """

    def _move_batch_to_device(self, batch: Tensor) -> Tensor:
        return batch.to(self.device, non_blocking=self.pin_memory)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int,
        global_bar: Optional[object] = None,
        total_steps: Optional[int] = None
    ) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_state = self._init_epoch_state()

        for batch_idx, batch in enumerate(loader):
            batch = self._move_batch_to_device(batch)
            if training:
                self.optimizer.zero_grad()

            context_manager = torch.enable_grad() if training else torch.no_grad()
            with context_manager:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)

                if training:
                    loss.backward()
                    self.optimizer.step()

            batch_size = int(batch.size(0))
            batch_metric = float(self.metric(outputs, batch)) if self.metric else 0.0
            self._update_epoch_state(
                epoch_state=epoch_state,
                batch_size=batch_size,
                loss=loss,
                metric_value=batch_metric
            )

            if training and self.rank == 0 and global_bar is not None and total_steps is not None:
                reduced_state = self._reduce_epoch_state(epoch_state)
                step = epoch * len(loader) + batch_idx + 1
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

        return self._reduce_epoch_state(epoch_state)

    def _run_validation_loop(self) -> Dict[str, float]:
        return self._run_epoch(
            loader=self.val_loader,
            training=False,
            epoch=0
        )

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        epoch = self.start_epoch
        try:
            for callback in self.callbacks:
                callback.on_train_begin(trainer=self)

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
                    def set_postfix(self, *args, **kwargs):
                        return None

                    def update(self, *args, **kwargs):
                        return None

                global_bar = _NoOpBar()

            for epoch in range(self.start_epoch, self.num_epochs):
                if self._is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch, trainer=self)

                self.model.train()
                if epoch == self.start_epoch and self._resume_batch_offset > 0 and self._resume_epoch_state is not None:
                    epoch_state = dict(self._resume_epoch_state)
                else:
                    epoch_state = self._init_epoch_state()
                last_eval_step = None
                should_stop = False
                resume_batch_offset = self._resume_batch_offset if epoch == self.start_epoch else 0

                for batch_idx, batch in enumerate(self.train_loader):
                    if batch_idx < resume_batch_offset:
                        continue

                    batch = self._move_batch_to_device(batch)
                    self.optimizer.zero_grad()

                    with torch.enable_grad():
                        outputs = self.model(batch)
                        loss = self.criterion(outputs, batch)
                        loss.backward()
                        self.optimizer.step()

                    batch_size = int(batch.size(0))
                    batch_metric = float(self.metric(outputs, batch)) if self.metric else 0.0
                    self._update_epoch_state(
                        epoch_state=epoch_state,
                        batch_size=batch_size,
                        loss=loss,
                        metric_value=batch_metric
                    )

                    step = epoch * len(self.train_loader) + batch_idx + 1
                    if self.rank == 0 and total_steps is not None:
                        reduced_state = self._reduce_epoch_state(epoch_state)
                        if step % self.logging_steps == 0 or step == total_steps:
                            tqdm.write(
                                f"step: {step}/{total_steps} | "
                                f"train_loss: {reduced_state['loss']:.4f} | "
                                f"train_metric: {reduced_state['metric']:.4f}"
                            )

                        global_bar.set_postfix({
                            'epoch': f"{epoch + 1}/{self.num_epochs}",
                            'avg_loss': f"{reduced_state['loss']:.4f}",
                            'avg_metric': f"{reduced_state['metric']:.4f}"
                        })

                    global_bar.update(1)

                    if self._should_run_step_validation(step=step, total_steps=total_steps):
                        train_metrics = self._reduce_epoch_state(epoch_state)
                        val_metrics = self._run_validation_loop()
                        epoch_progress = epoch + ((batch_idx + 1) / max(len(self.train_loader), 1))
                        last_logs = self._finalize_validation_event(
                            epoch_index=epoch,
                            epoch_progress=epoch_progress,
                            global_step=step,
                            epoch_state=epoch_state,
                            train_metrics=train_metrics,
                            val_metrics=val_metrics
                        )
                        last_eval_step = step
                        should_stop = self._run_validation_callbacks(epoch=epoch, logs=last_logs)
                        if should_stop:
                            break

                if should_stop:
                    break

                self._resume_batch_offset = 0
                self._resume_epoch_state = None
                train_metrics = self._reduce_epoch_state(epoch_state)
                if self.eval_strategy == 'epoch':
                    val_metrics = self._run_validation_loop()
                    last_logs = self._finalize_validation_event(
                        epoch_index=epoch,
                        epoch_progress=epoch + 1,
                        global_step=(epoch + 1) * len(self.train_loader),
                        epoch_state=epoch_state,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics
                    )
                    should_stop = self._run_validation_callbacks(epoch=epoch, logs=last_logs)

                if should_stop:
                    break

            for callback in self.callbacks:
                callback.on_train_end(trainer=self)
        except KeyboardInterrupt:
            if self.rank == 0:
                print(f"\nTraining interrupted at epoch {epoch + 1}.")

            cleanup_ddp()

        return self.history, self.model

    @torch.no_grad()
    def evaluate(
        self,
        plot: Optional[Union[Callable, List[Callable]]] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")

        self.model.eval()
        epoch_state = self._init_epoch_state()
        local_true = []
        local_pred = []

        for batch in self.test_loader:
            batch = self._move_batch_to_device(batch)
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch)
            batch_size = int(batch.size(0))
            batch_metric = float(self.metric(outputs, batch)) if self.metric else 0.0
            self._update_epoch_state(
                epoch_state=epoch_state,
                batch_size=batch_size,
                loss=loss,
                metric_value=batch_metric
            )

            local_true.append(batch.detach().cpu().numpy())
            local_pred.append(torch.sigmoid(outputs).detach().cpu().numpy())

        reduced_metrics = self._reduce_epoch_state(epoch_state)
        y_true = np.concatenate(local_true, axis=0) if local_true else np.empty((0,))
        y_pred = np.concatenate(local_pred, axis=0) if local_pred else np.empty((0,))

        if self._is_distributed:
            gathered_true = [None for _ in range(self.world_size)]
            gathered_pred = [None for _ in range(self.world_size)]
            all_gather_object(gathered_true, y_true)
            all_gather_object(gathered_pred, y_pred)
            y_true = np.concatenate(gathered_true, axis=0) if gathered_true else np.empty((0,))
            y_pred = np.concatenate(gathered_pred, axis=0) if gathered_pred else np.empty((0,))

        self.last_eval_metrics = dict(reduced_metrics)

        if self.rank == 0:
            print(
                f"test_loss: {reduced_metrics['loss']:.4f} | "
                f"test_metric: {reduced_metrics['metric']:.4f}"
            )
            if plot is not None:
                if isinstance(plot, list):
                    for index, visualizer in enumerate(plot):
                        output_path = os.path.join(self.outputs_dir, f'{self.run_name}_viz_{index + 1}.png')
                        visualizer(y_true, y_pred, save_fig=output_path if self.save_fig else None)
                else:
                    output_path = os.path.join(self.outputs_dir, f'{self.run_name}.png')
                    plot(y_true, y_pred, save_fig=output_path if self.save_fig else None)

        return reduced_metrics['loss'], y_true, y_pred
