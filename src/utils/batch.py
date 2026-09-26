from collections.abc import Mapping
from typing import Any, Callable

from torch import Tensor


def move_batch(
    batch: Any,
    move_tensor: Callable[[Tensor], Tensor],
    move_nested: Callable[[Any], Any],
    preserve_mappings: bool
) -> Any:
    if isinstance(batch, Tensor):
        return move_tensor(batch)

    if isinstance(batch, Mapping if preserve_mappings else dict):
        values = ((key, move_nested(value)) for key, value in batch.items())
        return type(batch)(values) if preserve_mappings else dict(values)

    if isinstance(batch, tuple):
        return tuple(move_nested(value) for value in batch)

    if isinstance(batch, list):
        return [move_nested(value) for value in batch]

    return batch
