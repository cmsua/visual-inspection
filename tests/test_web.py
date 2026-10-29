import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pytest
import yaml

from src.configs import InspectionConfig
from src.utils import InspectionResults
from web.app import InspectionApp
from web.server import InspectionRequestHandler


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

    def run(command: List[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured.update(kwargs)
        captured['settings'] = yaml.safe_load(Path(kwargs['env']['INSPECTION_CONFIG']).read_text())
        captured['command'] = command
        return subprocess.CompletedProcess(command, 0, stdout='ok', stderr='')

    monkeypatch.setattr('web.app.inspection_app.validate_calibration', lambda config: None)
    monkeypatch.setattr('web.app.inspection_app.subprocess.run', run)
    status, payload = request(app, 'POST', '/api/run', {})
    assert status == 200 and payload['status'] == 'ok'
    assert captured['settings']['inspection']['checkpoint_path'] == app.config.checkpoint_path
    assert captured['env']['INSPECTION_PYTHON']


def test_job_failures_reach_client(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, 'run_inspection_job', lambda path: subprocess.CompletedProcess([], 1, '', 'failed'))
    status, payload = request(app, 'POST', '/api/run', {})
    assert status == 500 and payload['stderr'] == 'failed'


def test_manual_load_accepts_repo_data_without_inference(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(app.config.data_root)
    data = root / 'data'
    data.mkdir()
    board = data / 'another.npy'
    np.save(board, np.zeros((8, 5, 12, 14, 3), dtype=np.uint8))
    app.config.data_root = str(data)
    monkeypatch.setattr('web.app.inspection_app.REPO_ROOT', root)
    monkeypatch.setattr(app, 'get_inspection', lambda *args: pytest.fail("Manual loading must not invoke inference"))
    status, payload = request(app, 'GET', '/api/board?path=./data/another.npy&inspect=false')
    assert status == 200
    assert payload['inspection']['metadata']['inspection_status'] == 'not_run'
    assert payload['inspection']['pw_flags'][2][2] is None
    assert payload['inspection']['pw_flags'][0][0] == -1
    assert request(app, 'GET', '/api/segment?path=./data/another.npy&row=2&col=2')[0] == 200
    assert request(app, 'GET', '/api/board?path=./data/../board.npy&inspect=false')[0] == 400


def test_calibration_blocker_has_recovery_command_without_starting_job(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(config: InspectionConfig) -> None:
        raise ValueError("Baseline, checkpoint, or skip mask changed. Recalibrate before inspection.")

    monkeypatch.setattr('web.app.inspection_app.validate_calibration', reject)
    monkeypatch.setattr('web.app.inspection_app.subprocess.run', lambda *args, **kwargs: pytest.fail("Blocked inspection started"))
    status, payload = request(app, 'POST', '/api/run', {})
    assert status == 400
    assert 'scripts.calibrate' in payload['error']
    assert app.config.source_path in payload['error']
    assert 'Manual review remains available' in payload['error']
    assert 'Traceback' not in payload['error']


def test_switching_boards_and_missing_path_diagnostics(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(app.config.data_root)
    first = Path(app.config.board_path)
    second = root / 'other_folder' / 'second.npy'
    second.parent.mkdir()
    np.save(second, np.zeros((8, 5, 12, 14, 3), dtype=np.uint8))
    monkeypatch.setattr('web.app.inspection_app.REPO_ROOT', root)
    monkeypatch.setattr(app, 'get_inspection', lambda *args: pytest.fail("Manual loading invoked inspection"))
    for path in (first.name, 'other_folder/second.npy', first.name):
        status, payload = request(app, 'GET', f'/api/board?path={path}&inspect=false')
        assert status == 200
        assert payload['grid_shape'] == {'rows': 8, 'cols': 5}

    for method, route, body in (
        ('GET', '/api/board?path=second.npy&inspect=false', None),
        ('POST', '/api/run', {'path': 'second.npy'})
    ):
        status, payload = request(app, method, route, body)
        assert status == 404
        assert str(root / 'second.npy') in payload['error']
        assert 'other_folder/second.npy' in payload['error']
        assert 'Matching filenames' in payload['error']


def test_missing_support_file_is_identified(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app.config.skipped_segments_path = str(Path(app.config.data_root) / 'missing-skips.json')
    status, payload = request(app, 'GET', '/api/board?inspect=false')
    assert status == 404
    assert 'missing-skips.json' in payload['error']


def test_manual_preview_remains_available_during_agent_inspection(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, 'get_inspection', lambda *args: pytest.fail("Preview invoked inference"))
    with app.execution_guard():
        status, payload = request(app, 'GET', '/api/board?inspect=false')
        assert status == 200
        assert payload['inspection']['metadata']['inspection_status'] == 'not_run'
        assert payload['inspection']['pw_flags'][2][2] is None
        assert request(app, 'GET', '/api/segment?row=2&col=2')[0] == 200
        assert request(app, 'GET', '/api/board')[0] == 400


def test_board_picker_lists_designated_folder_without_loading_tensors(app: InspectionApp, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(app.config.data_root)
    folder = root / 'bad'
    nested = folder / 'batch_02'
    nested.mkdir(parents=True)
    (folder / '001.npy').touch()
    (nested / '002.NPY').touch()
    (folder / 'notes.txt').touch()
    (folder / 'outside.npy').symlink_to('/etc/passwd')
    monkeypatch.setattr('web.app.inspection_app.REPO_ROOT', root)
    monkeypatch.setattr(app, '_load_board', lambda *args: pytest.fail('Board listing loaded a tensor'))
    monkeypatch.setattr(app, 'get_inspection', lambda *args: pytest.fail('Board listing ran inference'))
    with app.execution_guard():
        status, payload = request(app, 'GET', '/api/boards')

    assert status == 200
    assert Path(payload['folder']).name == 'bad'
    assert [board['name'] for board in payload['boards']] == ['001.npy', 'batch_02/002.NPY']
    assert app.resolve_board_path(payload['boards'][0]['path']) == folder / '001.npy'
    assert request(app, 'GET', '/api/boards?folder=bad/batch_02')[1]['boards'][0]['name'] == '002.NPY'
    assert request(app, 'GET', '/api/boards?folder=../')[0] == 400
    assert request(app, 'GET', '/api/boards?folder=missing')[0] == 404
    empty = root / 'incoming'
    empty.mkdir()
    assert request(app, 'GET', '/api/boards?folder=incoming')[1]['boards'] == []


def test_board_picker_defaults_to_data_root_without_bad_folder(app: InspectionApp) -> None:
    status, payload = request(app, 'GET', '/api/boards')
    assert status == 200
    assert [board['name'] for board in payload['boards']] == ['board.npy']
