import asyncio
import json
import shlex
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Generator, List, Tuple

import pytest
import yaml

from src.utils import InspectionResults
from web.agent import ChatSession, InspectionGateway
from web.app import ChatApp
from web.configs import AgentConfig
from web.server import validate_origin


@pytest.fixture
def chat(tmp_path: Path) -> Tuple[ChatApp, Dict, Path]:
    board = tmp_path / 'board.npy'
    board.touch()
    state = tmp_path / 'state.yaml'
    state.write_text(yaml.safe_dump({
        'visual_inspection_data_roots': [str(tmp_path)],
        'visual_inspection_artifact_root': str(tmp_path / 'runs')
    }))
    config = AgentConfig(
        tmp_path,
        Path(sys.executable),
        state,
        tmp_path / 'config.yaml',
        board,
        tmp_path / 'outputs'
    )

    @contextmanager
    def guard() -> Generator[None, None, None]:
        yield

    app = SimpleNamespace(
        resolve_board_path=lambda value: Path(value).resolve(),
        execution_guard=guard,
        build_board_payload=lambda path, inspection: {'inspection': inspection.to_dict()}
    )
    service = ChatApp(app, config)
    session = service.create(str(board))
    snapshot = tmp_path / 'result.json'
    result = InspectionResults.from_segment_flags(
        (8, 5),
        {(2, 3)},
        {(2, 3)},
        {(0, 0)},
        'reference.npy',
        str(board)
    )
    result.save_json(str(snapshot))
    entry = {
        'tool': 'ReadInspectionResult',
        'envelope': {'result': {'result_path': str(snapshot), 'freshness': {'status': 'unverified'}}}
    }
    service._capture(session, entry, {})
    service._save(session)
    return service, session, snapshot


def test_evidence_review_persists_without_inference_and_detects_changed_snapshot(chat: Tuple[ChatApp, Dict, Path]) -> None:
    service, session, snapshot = chat
    identity = session['results'][0]['result_id']
    evidence = service.evidence(session['id'], identity)
    assert evidence['review_context']['reviews'] == {}
    assert evidence['review_context']['freshness']['status'] == 'unverified'
    review = {
        'session_id': session['id'],
        'result_id': identity,
        'row': 2,
        'col': 3,
        'decision': 'uncertain',
        'reviewer': "Inspector"
    }
    assert service.review(review)['reviews']['2,3']['decision'] == 'uncertain'
    restarted = ChatApp(service.app, service.config)
    assert restarted.evidence(session['id'], identity)['review_context']['reviews']['2,3']['reviewer'] == "Inspector"
    with pytest.raises(ValueError, match='Skipped'):
        service.review({**review, 'row': 0, 'col': 0})

    snapshot.write_text(snapshot.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        service.evidence(session['id'], identity)


def test_snapshot_rejects_wrong_board_and_inconsistent_flags(chat: Tuple[ChatApp, Dict, Path]) -> None:
    service, session, snapshot = chat
    with pytest.raises(ValueError, match='different board'):
        service._validated_snapshot(snapshot, '/another/board.npy')

    document = json.loads(snapshot.read_text())
    document['hybrid_flags'][2][3] = 0
    snapshot.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='inconsistent'):
        service._validated_snapshot(snapshot, session['board'])


def test_authorization_paths_and_review_validation(chat: Tuple[ChatApp, Dict, Path]) -> None:
    service, session, _ = chat
    for action in ('inspect', 'refresh'):
        with pytest.raises(ValueError, match='authorize'):
            service.submit({'session_id': session['id'], 'action': action, 'message': "Inspect"})

    with pytest.raises(ValueError, match='within configured'):
        service._allowed_result('/etc/passwd')

    with pytest.raises(ValueError, match='conversation ID'):
        service.get('../state')

    with pytest.raises(ValueError, match='coordinates'):
        service.review({'row': True, 'col': 1})

    with pytest.raises(KeyError):
        service.artifact(session['id'], 'unregistered')


def test_job_workflow_captures_evidence_and_bounds_tools(chat: Tuple[ChatApp, Dict, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    service, session, snapshot = chat
    policies = []

    class FakeSession:
        def __init__(self, config: AgentConfig, policy: Dict, **kwargs: object) -> None:
            policies.append(policy)
            self.run_dir = config.output_dir / 'run'
            self.run_dir.mkdir(parents=True)
            self.report = {'evidence': [{
                'tool': 'InspectBoard',
                'envelope': {'result': {'result_path': str(snapshot), 'cache_status': 'reused'}}
            }]}

        def __enter__(self) -> 'FakeSession':
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def chat(self, message: str, history: List[Dict]) -> Dict:
            assert history[-1]['role'] == 'evidence'
            return {'answer': "One hybrid flag; review source (2, 3)."}

    monkeypatch.setattr('web.app.chat_app.ChatSession', FakeSession)
    job = service.submit({
        'session_id': session['id'],
        'action': 'inspect',
        'message': "Inspect",
        'authorize_inspection': True
    })
    service.jobs[job['id']]['thread'].join(timeout=5)
    result = service.get(session['id'])
    assert result['jobs'][0]['status'] == 'completed'
    assert result['messages'][-1]['role'] == 'assistant'
    assert policies[0]['tools'] == ['VisualInspectionPreflight', 'InspectBoard', 'ReadInspectionResult']
    assert policies[0]['board_path'] == session['board']
    assert result['results'][-1]['cache_status'] == 'reused'
    service.close()


def test_cross_site_and_rebinding_requests_rejected() -> None:
    for headers in (
        {'Host': 'evil.example'},
        {'Host': 'localhost:3000', 'Origin': 'https://evil.example'},
        {'Host': 'localhost:3000', 'Sec-Fetch-Site': 'cross-site'}
    ):
        with pytest.raises(ValueError):
            validate_origin(SimpleNamespace(headers=headers, server=SimpleNamespace(chat=True)))

    validate_origin(SimpleNamespace(
        headers={'Host': '127.0.0.1:3000', 'Origin': 'http://127.0.0.1:3000'},
        server=SimpleNamespace(chat=True)
    ))


def test_chat_notification_rejects_model_selected_arguments(chat: Tuple[ChatApp, Dict, Path]) -> None:
    service, _, _ = chat
    session = ChatSession(service.config, {'tools': ['InspectBoard']}, cancelled=threading.Event())
    with pytest.raises(RuntimeError, match='Unexpected tool'):
        session._notification({
            'method': 'item/started',
            'params': {'item': {
                'type': 'mcpToolCall',
                'server': 'heptapod_preflight',
                'tool': 'InspectBoard',
                'arguments': {'force': True}
            }}
        })


@pytest.mark.parametrize('action', ['inspect', 'refresh'])
def test_gateway_enforces_readiness_fixed_arguments_and_single_call(action: str) -> None:
    calls = []
    policy = {
        'action': action,
        'tools': ['VisualInspectionPreflight', 'InspectBoard', 'ReadInspectionResult'],
        'config_path': '/config.yaml',
        'board_path': '/board.npy',
        'upstream': {}
    }

    async def invoke(upstream: Dict, name: str, arguments: Dict) -> object:
        calls.append((name, arguments))
        envelope = {
            'schema_version': 'visual-inspection-bridge/1',
            'status': 'ok',
            'result': {
                'ready': True,
                'result_path': '/saved.json'
            }
        }
        return SimpleNamespace(is_error=False, content=[SimpleNamespace(type='text', text=json.dumps(envelope))])

    async def exercise() -> None:
        gateway = InspectionGateway(policy, invoke)
        with pytest.raises(ValueError, match='successful ready preflight'):
            await gateway.call('InspectBoard')

        with pytest.raises(ValueError, match='No saved'):
            await gateway.call('ReadInspectionResult')

        with pytest.raises(ValueError, match='not authorized'):
            await gateway.call('Calibrate')

        await gateway.call('VisualInspectionPreflight')
        await gateway.call('InspectBoard')
        await gateway.call('ReadInspectionResult')

        with pytest.raises(ValueError, match='already run'):
            await gateway.call('InspectBoard')

    asyncio.run(exercise())
    assert calls == [
        ('VisualInspectionPreflight', {'config_path': '/config.yaml', 'board_path': '/board.npy'}),
        ('InspectBoard', {'config_path': '/config.yaml', 'board_path': '/board.npy', 'force': action == 'refresh'}),
        ('ReadInspectionResult', {'result_path': '/saved.json'})
    ]


def test_summary_loads_selected_boards_saved_results(chat: Tuple[ChatApp, Dict, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    service, session, snapshot = chat
    board = Path(session['board'])
    board.with_suffix('.inspection.json').write_bytes(snapshot.read_bytes())
    session['results'] = []
    service._save(session)
    monkeypatch.setattr(service, '_run', lambda *args: None)
    job = service.submit({'session_id': session['id'], 'action': 'chat', 'message': "Summarize the results"})
    service.jobs[job['id']]['thread'].join(timeout=5)
    saved = service.get(session['id'])
    assert saved['results'][0]['tool'] == 'SavedInspection'
    assert saved['results'][0]['inspection']['hybrid_flags'][2][3] == 1
    assert saved['results'][0]['freshness']['status'] == 'unverified'


def test_chat_inspection_resolves_board_and_rejects_ambiguity(chat: Tuple[ChatApp, Dict, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    service, session, snapshot = chat
    root = snapshot.parent
    folder = root / 'bad_example'
    folder.mkdir()
    board = folder / '320XLF4CQH00445.npy'
    board.touch()
    service.app.config = SimpleNamespace(data_root=str(root), source_path=str(service.config.inspection_config))
    policies = []
    monkeypatch.setattr(service, '_run', lambda job, policy, *args: policies.append(policy))
    job = service.request({'message': "Inspect board 445", 'board_path': '/irrelevant.npy'})
    service.jobs[job['id']]['thread'].join(timeout=5)
    assert job['action'] == 'inspect'
    assert job['board_path'] == str(board)
    assert policies[0]['tools'] == ['VisualInspectionPreflight', 'InspectBoard', 'ReadInspectionResult']
    assert json.loads(policies[0]['runtime_state']['visual_inspection_writable_data_roots']) == [str(root)]
    service.jobs[job['id']]['status'] = 'completed'
    assert service.find_board('00445') == board
    (root / board.name).touch()
    with pytest.raises(ValueError, match='Multiple boards'):
        service.request({'message': "Inspect board 445"})

    with pytest.raises(ValueError, match='No board'):
        service.find_board('99999')

    assert service.find_board(str(board)) == board
    question = service.request({'message': "Do not inspect board 445", 'board_path': str(board)})
    service.jobs[question['id']]['thread'].join(timeout=5)
    assert question['action'] == 'chat'


def test_maintenance_help_uses_bound_config_without_authorizing_execution(chat: Tuple[ChatApp, Dict, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    service, _, _ = chat
    config_path = service.config.output_dir / 'config with spaces.yaml'
    session = ChatSession(
        service.config,
        {
            'action': 'chat',
            'tools': ['VisualInspectionPreflight'],
            'board_path': str(service.config.board_path),
            'config_path': str(config_path)
        }
    )
    captured = {}

    def answer(message: str, text: str, require_preflight: bool) -> Dict:
        captured.update(json.loads(text))
        assert not require_preflight
        return {'answer': "Run the documented calibration command from the repository root."}

    monkeypatch.setattr(session, '_ask_text', answer)
    session.chat("What is the command to recalibrate thresholds?", [])
    help_text = captured['maintenance_help']
    assert shlex.split(help_text['recalibration_command']) == [
        'venv/bin/python', '-m', 'scripts.calibrate', '--config-path', str(config_path)
    ]
    assert help_text['execution_authorized'] is False
    assert captured['authorized_action'] == 'chat'
    assert session.report['evidence'] == []


@pytest.mark.parametrize(
    ('message', 'expected'),
    [
        ("Inspect board 67", 67),
        ("Inspect the board with ID 67", 67),
        ("Could you please inspect the board with an ID of 67?", 67),
        ("Inspect a board with an ID higher than 500", 501),
        ("Please inspect any board with ID above 500.", 501),
        ("Inspect a board with ID >= 500", 500),
        ("Inspect a board with an ID at least 500", 500),
        ("Inspect a board with ID below 500", 67),
        ("Recompute the board with ID 67, please", 67)
    ]
)

def test_natural_inspection_requests_select_matching_board(
    chat: Tuple[ChatApp, Dict, Path],
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected: int
) -> None:
    service, _, snapshot = chat
    root = snapshot.parent
    boards = {}
    for folder, identity in (('train', 67), ('val', 500), ('bad', 501), ('test', 600)):
        directory = root / folder
        directory.mkdir()
        board = directory / f'320XLF4CQH{identity:05d}.npy'
        board.touch()
        boards[identity] = board

    service.app.config = SimpleNamespace(data_root=str(root), source_path=str(service.config.inspection_config))
    monkeypatch.setattr(service, '_run', lambda *args: None)
    job = service.request({'message': message})
    service.jobs[job['id']]['thread'].join(timeout=5)
    assert job['board_path'] == str(boards[expected])
    assert job['action'] == ('refresh' if message.startswith('Recompute') else 'inspect')
    with pytest.raises(ValueError, match='Multiple boards'):
        service.find_board('the board with ID above 500')

    with pytest.raises(ValueError, match='No board'):
        service.find_board('a board with ID higher than 600')

    duplicate = root / boards[501].name
    duplicate.touch()
    with pytest.raises(ValueError, match='Multiple boards'):
        service.find_board('a board with ID higher than 500')
