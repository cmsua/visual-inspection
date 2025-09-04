from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 20
    start_epoch: int = 0
    logging_dir: str = 'logs'
    logging_steps: int = 500
    save_best: bool = True
    save_ckpt: bool = True
    device: str = None
    num_workers: int = 0
    pin_memory: bool = False

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)