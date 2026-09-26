import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable


def file_signatures(paths: Iterable[str]) -> Dict:
    result = {}
    for value in paths:
        path = Path(value).resolve()
        stat = path.stat()
        result[str(path)] = [stat.st_size, stat.st_mtime_ns]

    return result


def save_json(path: str, payload: Dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=4, allow_nan=False)

        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
