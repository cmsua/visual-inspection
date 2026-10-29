import copy
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import pytest
import yaml

from web.agent import AgentError, CodexTransport, PreflightSession
from web.configs import AgentConfig


@pytest.fixture
def setup_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple:
    repo = tmp_path / 'heptapod'
    repo.mkdir()
    (repo / 'toolkit.yaml').write_text(yaml.safe_dump({'tools': [
        {'name': name + 'Tool', 'bundle': 'visual_inspection', 'module': 'tools.visual_inspection.agent_tools'}
        for name in ('VisualInspectionPreflight', 'InspectBoard', 'ReadInspectionResult')
    ]}))
    state = tmp_path / 'state.yaml'
    state.write_text(yaml.safe_dump({
        'visual_inspection_repo': str(tmp_path / 'upstream'),
        'visual_inspection_python': sys.executable,
        'visual_inspection_data_roots': json.dumps([str(tmp_path / 'data')]),
        'visual_inspection_artifact_root': str(repo / 'old_runs'),
        'visual_inspection_writable_data_roots': [],
        'unrelated_secret': 'must-not-be-forwarded'
    }))
    inspection = tmp_path / 'inspection.yaml'
    inspection.write_text('inspection: {}\n')
    config = AgentConfig(
        repo,
        Path(sys.executable),
        state,
        inspection,
        None,
        tmp_path / 'outputs'
    )
    peer = FakePeer(config)
    monkeypatch.setattr('web.agent.session.CodexTransport', lambda *args: peer)
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'codex'))
    return config, peer


class FakePeer:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.events = []
        self.requests = []
        self.sent = []
        self.closed = False
        self.counter = 0
        self.mode = 'normal'
        self.ready = False
        self.extra_tool = False
        self.completed_before_reply = False
        self.turn = 0

    def __enter__(self) -> 'FakePeer':
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def send(self, message: Dict) -> None:
        self.sent.append(message)

    def request(self, method: str, params: Dict) -> int:
        self.counter += 1
        self.requests.append((method, params))
        result = {}
        notifications = []
        if method == 'thread/start':
            result = {'thread': {'id': 'thread1'}, 'model': 'test-model', 'sandbox': {'type': 'readOnly'}}
            if self.mode == 'unsafe_sandbox':
                result['sandbox']['type'] = 'dangerFullAccess'
        elif method == 'mcpServerStatus/list':
            tools = {'VisualInspectionPreflight': {'name': 'VisualInspectionPreflight'}}
            if self.extra_tool:
                tools['InspectBoard'] = {'name': 'InspectBoard'}

            result = {'data': [{'name': 'heptapod_preflight', 'tools': tools}], 'nextCursor': None}
        elif method == 'turn/start':
            self.turn += 1
            turn = {'id': str(self.turn), 'status': 'failed' if self.mode == 'failed_turn' else 'completed'}
            result = {'turn': turn}
            if self.turn == 1 and self.mode != 'no_tool':
                if self.mode == 'approval':
                    notifications.append({'id': 'server1', 'method': 'item/commandExecution/requestApproval', 'params': {}})
                else:
                    item = {
                        'id': 'call1',
                        'type': 'mcpToolCall',
                        'server': 'heptapod_preflight',
                        'tool': 'VisualInspectionPreflight',
                        'arguments': {'config_path': str(self.config.inspection_config)},
                        'status': 'inProgress'
                    }
                    if self.mode == 'wrong_path':
                        item['arguments']['config_path'] = '/another/config.yaml'

                    notifications.append({'method': 'item/started', 'params': {'threadId': 'thread1', 'item': copy.deepcopy(item)}})
                    item['status'] = 'completed'
                    envelope = {
                        'schema_version': 'visual-inspection-bridge/1',
                        'status': 'ok',
                        'result': {
                            'schema_version': 'visual-inspection-preflight/1',
                            'ready': self.ready,
                            'checks': [{'name': 'calibration', 'status': 'blocked'}]
                        }
                    }
                    if self.mode == 'invalid_tool':
                        envelope = {'status': 'ok', 'result': {'ready': 'yes'}}

                    item['result'] = {'content': [{'type': 'text', 'text': json.dumps(envelope)}]}
                    notifications.append({'method': 'item/completed', 'params': {'threadId': 'thread1', 'item': item}})

            notifications.extend([
                {'method': 'item/completed', 'params': {'threadId': 'thread1', 'item': {'type': 'agentMessage', 'text': "Calibration blocks inspection."}}},
                {'method': 'turn/completed', 'params': {'threadId': 'thread1', 'turn': turn}}
            ])

        reply = {'id': self.counter, 'result': result}
        self.events.extend(notifications + [reply] if self.completed_before_reply else [reply] + notifications)
        return self.counter

    def receive(self, deadline: float, cancelled: threading.Event) -> Dict:
        if cancelled.is_set():
            raise AgentError("Cancelled")
        if not self.events:
            raise AgentError("Test peer has no pending response")

        return self.events.pop(0)


@pytest.mark.parametrize('early_completion', [False, True])
def test_preflight_blocker_is_valid_evidence_and_followup_reuses_thread(setup_agent: Tuple, early_completion: bool) -> None:
    config, peer = setup_agent
    peer.completed_before_reply = early_completion
    original = config.state_config.read_bytes()
    with PreflightSession(config) as session:
        first = session.ask("Check readiness")
        second = session.ask("Why?", require_preflight=False)
        assert first['answer'] == second['answer']
        assert session.report['preflight']['result']['ready'] is False
        assert session.tool_count == 1
        report_path = session.run_dir / 'run.json'

    assert peer.closed
    assert config.state_config.read_bytes() == original
    report = json.loads(report_path.read_text())
    assert report['status'] == 'completed' and len(report['turns']) == 2
    assert {params['threadId'] for method, params in peer.requests if method == 'turn/start'} == {'thread1'}
    assert not list(config.heptapod_repo.rglob('*.inspection.json'))


@pytest.mark.parametrize('mode', ['no_tool', 'invalid_tool', 'wrong_path', 'approval', 'failed_turn', 'unsafe_sandbox'])
def test_unverified_or_disallowed_execution_fails_and_retains_diagnostics(setup_agent: Tuple, mode: str) -> None:
    config, peer = setup_agent
    peer.mode = mode
    session = PreflightSession(config)
    with pytest.raises(AgentError):
        with session:
            session.ask("Check readiness")

    assert peer.closed
    assert json.loads((session.run_dir / 'run.json').read_text())['status'] == 'failed'
    if mode == 'approval':
        assert {'id': 'server1', 'result': {'decision': 'decline'}} in peer.sent


def test_discovery_rejects_inspection_tool_before_any_model_turn(setup_agent: Tuple) -> None:
    config, peer = setup_agent
    peer.extra_tool = True
    with pytest.raises(AgentError, match='exactly one'):
        with PreflightSession(config):
            pytest.fail("Unsafe inventory accepted")

    assert peer.closed
    assert not any(method == 'turn/start' for method, _ in peer.requests)


def test_server_config_exposes_only_preflight_and_redirects_artifacts(setup_agent: Tuple) -> None:
    config, _ = setup_agent
    run_dir = config.output_dir / 'test'
    server = config.server_config(run_dir)
    args = server['args']
    specs = json.loads(args[args.index('--tools-spec') + 1])
    assert [entry['name'] for entry in specs] == ['VisualInspectionPreflightTool']
    state = json.loads(args[args.index('--state-config') + 1])
    assert state['visual_inspection_artifact_root'] == str(run_dir / 'heptapod')
    assert 'unrelated_secret' not in state
    assert state['visual_inspection_writable_data_roots'] == []
    assert server['enabled_tools'] == ['VisualInspectionPreflight']


def test_reject_artifact_output_inside_heptapod(setup_agent: Tuple) -> None:
    config, _ = setup_agent
    with pytest.raises(AgentError, match='outside HEPTAPOD'):
        with PreflightSession(replace(config, output_dir=config.heptapod_repo / 'outputs')):
            pytest.fail("Write into HEPTAPOD accepted")


def test_transport_timeout_disconnect_and_cancellation_cleanup(tmp_path: Path) -> None:
    sleeper = [sys.executable, '-u', '-c', 'import time; time.sleep(60)']
    with CodexTransport(sleeper, tmp_path, tmp_path / 'timeout.log') as transport:
        process = transport.process
        with pytest.raises(AgentError, match='timed out'):
            transport.receive(time.monotonic() + 0.05)

        cancelled = threading.Event()
        cancelled.set()
        with pytest.raises(AgentError, match='cancelled'):
            transport.receive(time.monotonic() + 1, cancelled)

    assert process.poll() is not None
    with CodexTransport([sys.executable, '-c', 'pass'], tmp_path, tmp_path / 'exit.log') as transport:
        with pytest.raises(AgentError, match='disconnected'):
            transport.receive(time.monotonic() + 3)


def test_transport_rejects_invalid_protocol_output(tmp_path: Path) -> None:
    command = [sys.executable, '-u', '-c', 'print("not json")']
    with CodexTransport(command, tmp_path, tmp_path / 'invalid.log') as transport:
        with pytest.raises(AgentError, match='Invalid Codex output'):
            transport.receive(time.monotonic() + 3)


def test_transport_cleans_up_worker_in_separate_session(tmp_path: Path) -> None:
    child_code = 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'
    parent_code = (
        'import json, subprocess, sys, time; '
        f'child = subprocess.Popen([sys.executable, "-c", {child_code!r}], start_new_session=True); '
        'print(json.dumps({"child": child.pid}), flush=True); time.sleep(60)'
    )
    with CodexTransport([sys.executable, '-u', '-c', parent_code], tmp_path, tmp_path / 'tree.log') as transport:
        child_pid = transport.receive(time.monotonic() + 3)['child']
        assert Path(f'/proc/{child_pid}/stat').exists()

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            state = Path(f'/proc/{child_pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            return

        if state == 'Z':
            return

        time.sleep(0.01)

    pytest.fail("Detached worker still running after cleanup")


def test_yaml_paths_preserve_venv_symlink_and_timeout_validation(tmp_path: Path) -> None:
    interpreter = tmp_path / 'venv-python'
    interpreter.symlink_to(sys.executable)
    values = {
        'heptapod_repo': '.',
        'heptapod_python': 'venv-python',
        'state_config': 'state.yaml',
        'inspection_config': 'inspection.yaml',
        'output_dir': 'outputs'
    }
    path = tmp_path / 'agent.yaml'
    path.write_text(yaml.safe_dump({'agent_preflight': values}))
    config = AgentConfig.from_yaml(path)
    assert config.heptapod_python == interpreter
    assert config.output_dir == tmp_path / 'outputs'
    values['timeout_seconds'] = float('inf')
    path.write_text(yaml.safe_dump({'agent_preflight': values}))
    with pytest.raises(ValueError, match='timeout'):
        AgentConfig.from_yaml(path)
