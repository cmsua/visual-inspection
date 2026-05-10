from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class TrainConfig:
    batch_size: int = 8
    criterion: Dict = field(default_factory=lambda: {'name': 'cross_entropy_loss', 'kwargs': {}})
    optimizer: Dict = field(default_factory=lambda: {'name': 'adam', 'kwargs': {'lr': 1e-3, 'eps': 1e-4}})
    optimizer_wrapper: Dict = None
    scheduler: Dict = None
    callbacks: List = None
    num_epochs: int = 20
    start_epoch: int = 0
    logging_dir: str = 'logs'
    logging_steps: int = 40
    eval_strategy: str = 'epoch'
    eval_steps: int = 0
    progress_bar: bool = True
    best_model_monitor: str = 'val_loss'
    best_model_mode: str = 'min'
    save_best: bool = True
    save_ckpt: bool = True
    save_fig: bool = False
    num_workers: int = 0
    pin_memory: bool = False

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)
