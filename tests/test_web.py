import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import pytest
import yaml

from src.configs.inspection_config import InspectionConfig
from src.utils.get_results import InspectionResults
from web.app.inspection_app import InspectionApp
from web.server.handlers import InspectionRequestHandler


class MemorySocket:
    def __init__(self, request: bytes) -> None:
        self.request = io.BytesIO(request)
        self.response = io.BytesIO()

    def makefile(self, *args: object, **kwargs: object) -> io.BytesIO:
        return self.request

    def sendall(self, data: bytes) -> None:
        self.response.write(data)


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InspectionApp:
    board = tmp_path / 'board.npy'
    np.save(board, np.zeros((8, 5, 12, 14, 3), dtype=np.uint8))
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps({'files': {}, 'checkpoints': {}, 'version': 1}))
    config = InspectionConfig(data_root=str(tmp_path), board_path=str(board), labels_path=str(labels)).resolve()
    config.source_path = str(tmp_path / 'inspection.yaml')
    application = InspectionApp(config)
    monkeypatch.setattr(application, '_load_board', lambda path: np.load(path))
    result = InspectionResults.from_segment_flags(
        (8, 5),
        [(1, 1)],
        [(1, 1)],
        {(0, 0)},
        'baseline',
        str(board),
        (12, 14, 3)
    )
    monkeypatch.setattr(application, 'get_inspection', lambda path: result)
    return application


def request(app: InspectionApp, method: str, path: str, payload: Dict = None) -> Tuple[int, Dict]:
    body = json.dumps(payload).encode() if payload is not None else b''
    raw = f"{method} {path} HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
    connection = MemorySocket(raw)
    InspectionRequestHandler(connection, ('127.0.0.1', 12345), SimpleNamespace(app=app))
    headers, content = connection.response.getvalue().split(b'\r\n\r\n', 1)
    return int(headers.split()[1]), json.loads(content)


def test_http_board_segment_and_label_round_trip(app: InspectionApp) -> None:
    status, board = request(app, 'GET', '/api/board')
    assert status == 200
    assert board['grid_shape'] == {'rows': 8, 'cols': 5}
    assert board['inspection']['hybrid_flags'][1][1] == 1
    status, segment = request(app, 'GET', '/api/segment?row=1&col=1')
    assert status == 200 and segment['image'].startswith('iVBOR')
    status, labels = request(app, 'POST', '/api/label', {'row': 1, 'col': 1, 'damaged': True})
    assert status == 200 and labels['damaged_segments'] == [{'row': 1, 'col': 1}]
    reloaded = json.loads(Path(app.config.labels_path).read_text())
    assert any(entry['damaged'] == [{'row': 1, 'col': 1}] for entry in reloaded['files'].values())


def test_http_rejects_out_of_bounds_and_non_boolean_labels(app: InspectionApp) -> None:
    assert request(app, 'GET', '/api/segment?row=8&col=0')[0] == 400
    assert request(app, 'POST', '/api/label', {'row': 1, 'col': 1, 'damaged': 'false'})[0] == 400
    assert request(app, 'POST', '/api/label', {'row': -1, 'col': 0})[0] == 400
    assert request(app, 'GET', '/api/board?path=/etc/passwd')[0] == 400


def test_job_uses_server_python_and_yaml(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        captured.update(kwargs)
        captured['settings'] = yaml.safe_load(Path(kwargs['env']['INSPECTION_CONFIG']).read_text())
        captured['command'] = command
        return subprocess.CompletedProcess(command, 0, stdout='ok', stderr='')

    monkeypatch.setattr('web.app.inspection_app.subprocess.run', run)
    status, payload = request(app, 'POST', '/api/run', {})
    assert status == 200 and payload['status'] == 'ok'
    assert captured['settings']['inspection']['checkpoint_path'] == app.config.checkpoint_path
    assert captured['env']['INSPECTION_PYTHON']


def test_job_failures_reach_client(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, 'run_inspection_job', lambda path: subprocess.CompletedProcess([], 1, '', 'failed'))
    status, payload = request(app, 'POST', '/api/run', {})
    assert status == 500 and payload['stderr'] == 'failed'
