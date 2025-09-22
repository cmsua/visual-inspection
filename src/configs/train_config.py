from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    batch_size: int = 8
    criterion: dict = field(default_factory=lambda: {'name': 'cross_entropy_loss', 'kwargs': {}})
    optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'kwargs': {'lr': 1e-3, 'eps': 1e-4}})
    scheduler: dict = field(default_factory=lambda: {'name': 'exponential_lr', 'kwargs': {'gamma': 0.96}})
    callbacks: list = field(default_factory=lambda: [{'name': 'early_stopping', 'kwargs': {'patience': 5}}])
    num_epochs: int = 50
    start_epoch: int = 0
    logging_dir: str = 'logs'
    logging_steps: int = 25
    progress_bar: bool = True
    save_best: bool = True
    save_ckpt: bool = True
    save_fig: bool = False
    num_workers: int = 0
    pin_memory: bool = False

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)