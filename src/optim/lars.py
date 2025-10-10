from typing import List, Dict
from contextlib import contextmanager

from torch.optim import Optimizer


class LARS(Optimizer):
    """
    PyTorch implementation of the LARS optimizer wrapper.

    Parameters
    ----------
    optimizer: Optimizer
        The inner optimizer.
    eps: float
        Term added to the denominator to improve numerical stability.
    eta: float
        LARS coefficient as used in the LARS paper.

    .. References::
        Yang You, Igor Gitman, and Boris Ginsburg.
        [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888).
        *arXiv preprint arXiv:1708.03888*, 2017.

        Chunmyong Park, Heungsub Lee, Myungryong Jeong, Woonhyuk Baek, and Chiheon Kim.
        [torchlars, A LARS implementation in PyTorch](https://github.com/kakaobrain/torchlars).
        *GitHub*, 2019.
    """
    def __init__(self, optimizer: Optimizer, eps: float = 1e-8, eta: float = 0.001):
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if eta < 0.0:
            raise ValueError(f"Invalid trust coefficient: {eta}")
        
        self.optim = optimizer
        self.eps = eps
        self.eta = eta

    def __getstate__(self):
        state = {
            'eps': self.eps,
            'eta': self.eta,
            'optim_state': self.optim.state_dict()
        }
        
        return state

    def __setstate__(self, state: Dict):
        self.eps = state['eps']
        self.eta = state['eta']
        self.optim.load_state_dict(state['optim_state'])

    @property
    def param_groups(self) -> Dict:
        return self.optim.param_groups

    def state_dict(self) -> Dict:
        return self.optim.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group: Dict):
        self.optim.add_param_group(param_group)

    @contextmanager
    def hide_weight_decays(self):
        # Temporarily set weight_decay to 0 in each param group and yield the original values
        weight_decays = []
        for group in self.optim.param_groups:
            wd = group.get('weight_decay', 0.0)
            weight_decays.append(wd)
            group['weight_decay'] = 0.0

        try:
            yield weight_decays
        finally:
            for group, wd in zip(self.optim.param_groups, weight_decays):
                group['weight_decay'] = wd

    def compute_adaptive_lr(self, param_norm: float, grad_norm: float, weight_decay: float) -> float:
        return self.eta * param_norm / (grad_norm + weight_decay * param_norm + self.eps)

    def apply_adaptive_lrs(self, weight_decays: List):
        # Iterate over each parameter group and scale its gradients.
        for group, weight_decay in zip(self.optim.param_groups, weight_decays):
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_norm = p.data.norm(2)
                grad_norm = grad.norm(2)
                if param_norm > 0 and grad_norm > 0:
                    adaptive_lr = self.compute_adaptive_lr(param_norm, grad_norm, weight_decay)
                else:
                    adaptive_lr = 1.0

                # Apply weight decay: p.grad = p.grad + weight_decay * p.data
                p.grad.data.add_(p.data, alpha=weight_decay)

                # Scale gradient by the computed adaptive learning rate
                p.grad.data.mul_(adaptive_lr)

    def step(self, *args, **kwargs) -> float:
        # First hide weight decays, apply adaptive learning rates, then call the base optimizer
        with self.hide_weight_decays() as weight_decays:
            self.apply_adaptive_lrs(weight_decays)
            
            return self.optim.step(*args, **kwargs)