import copy
import hashlib
import json
import operator
import re
import sys
import threading
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import yaml

from src.configs import REPO_ROOT
from src.utils import InspectionResults

from ..agent import ChatSession
from ..configs import AgentConfig


class ChatApp:
    def __init__(self, app: object, config: AgentConfig) -> None:
        self.app = app
        self.config = config
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False
        state = yaml.safe_load(config.state_config.read_text())
        roots = state['visual_inspection_data_roots']
        roots = json.loads(roots) if isinstance(roots, str) else roots
        self.result_roots = [Path(root).resolve() for root in roots]
        if hasattr(app, 'config'):
            self.result_roots.append(Path(app.config.data_root).resolve())

        self.result_roots.extend([config.output_dir.resolve(), Path(state['visual_inspection_artifact_root']).resolve()])

    def _path(self, identity: str) -> Path:
        if not isinstance(identity, str) or len(identity) != 32 or any(char not in '0123456789abcdef' for char in identity):
            raise ValueError("Invalid conversation ID.")

        return self.config.output_dir / 'conversations' / (identity + '.json')

    def _load(self, identity: str) -> Dict:
        return json.loads(self._path(identity).read_text())

    def _save(self, session: Dict) -> None:
        path = self._path(session['id'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(session, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def status(self) -> Dict:
        folder = self.config.output_dir / 'conversations'
        sessions = []
        with self.lock:
            for path in sorted(folder.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
                document = json.loads(path.read_text())
                sessions.append({'id': document['id'], 'board': document['board']})

        return {'enabled': True, 'sessions': sessions, 'config_path': str(self.config.inspection_config)}

    def create(self, board: str) -> Dict:
        path = self.app.resolve_board_path(board)
        session = {
            'id': uuid.uuid4().hex,
            'board': str(path),
            'messages': [],
            'results': [],
            'artifacts': {},
            'reviews': {}
        }
        with self.lock:
            self._save(session)

        return session

    def get(self, identity: str) -> Dict:
        with self.lock:
            result = self._load(identity)
            result['jobs'] = [self._public(job) for job in self.jobs.values() if job['session_id'] == identity]
            return result

    def _public(self, job: Dict) -> Dict:
        return {key: value for key, value in job.items() if key not in ('cancel', 'thread')}

    def request(self, payload: Dict) -> Dict:
        message = payload.get('message', '')
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= 8000:
            raise ValueError("Enter a message of at most 8000 characters.")

        command = re.fullmatch(
            r'(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?'
            r'(inspect|reinspect|recompute)\s+(.+?)(?:,?\s+please)?[.!?]?',
            message.strip(),
            re.IGNORECASE
        )
        action = 'chat'
        board = payload.get('board_path')
        if command:
            action = 'inspect' if command[1].lower() == 'inspect' else 'refresh'
            board = str(self.find_board(command[2].strip().strip('\"\'')))

        with self.lock:
            if any(job['status'] in ('queued', 'running') for job in self.jobs.values()):
                raise ValueError("An agent request is already running. Wait or stop it first.")

            resolved = self.app.resolve_board_path(board)
            session = self._load(payload['session_id']) if payload.get('session_id') else None
            if session is None or session['board'] != str(resolved):
                session = self.create(str(resolved))

            return self.submit({
                'session_id': session['id'],
                'message': message,
                'action': action,
                'authorize_inspection': command is not None
            })

    def find_board(self, query: str) -> Path:
        original = query
        choose_any = re.match(r'^(?:a|an|any)\s+board\b', query, re.IGNORECASE) is not None
        query = re.sub(
            r'^(?:(?:the|a|an|any)\s+)?board\s+(?:(?:with|having)\s+)?'
            r'(?:(?:a|an|the)\s+)?(?:id\s*)?(?:(?:of|is|number)\s+|#\s*)?',
            '',
            query,
            flags=re.IGNORECASE
        ).strip().strip('\"\'')
        query = re.sub(r'^id\s*', '', query, flags=re.IGNORECASE)
        if '/' in query or '\\' in query:
            return self.app.resolve_board_path(query)

        comparisons = {
            '>': operator.gt,
            'higher than': operator.gt,
            'greater than': operator.gt,
            'above': operator.gt,
            'over': operator.gt,
            '>=': operator.ge,
            'at least': operator.ge,
            '<': operator.lt,
            'lower than': operator.lt,
            'less than': operator.lt,
            'below': operator.lt,
            'under': operator.lt,
            '<=': operator.le,
            'at most': operator.le
        }
        condition = re.fullmatch(r'(.+?)\s*(\d+)', query.lower())
        compare = comparisons.get(condition[1].strip()) if condition else None
        root = Path(self.app.config.data_root).resolve()
        matches = {}
        for path in root.rglob('*.npy'):
            suffix = re.search(r'(\d+)$', path.stem)
            identity = int(suffix[1]) if suffix else None
            matched = path.stem == query or path.name == query
            if identity is not None:
                if query.isdigit():
                    matched = matched or identity == int(query)
                elif compare is not None:
                    matched = compare(identity, int(condition[2]))

            if matched and path.is_file() and path.resolve().is_relative_to(root):
                matches[path.resolve()] = identity

        if not matches:
            raise ValueError(
                f"No board matches `{original}` in `{root}`. "
                "Use a filename, an exact ID, or a comparison such as 'a board with an ID higher than 500'."
            )
        if choose_any and compare is not None:
            # Choose the lowest matching ID; duplicate scans still require a full path.
            lowest = min(matches.values())
            matches = {path: identity for path, identity in matches.items() if identity == lowest}

        paths = sorted(matches)
        if len(paths) > 1:
            choices = ', '.join(str(path) for path in paths)
            raise ValueError(f"Multiple boards match `{original}`: {choices}. Send Inspect followed by one full path.")

        return paths[0]

    def submit(self, payload: Dict) -> Dict:
        action = payload.get('action', 'chat')
        if action not in ('chat', 'preflight', 'inspect', 'refresh', 'read'):
            raise ValueError("Unknown chat action.")

        message = payload.get('message', '')
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= 8000:
            raise ValueError("Enter a message of at most 8000 characters.")
        if action in ('inspect', 'refresh') and payload.get('authorize_inspection') is not True:
            raise ValueError("Select Inspect or Recompute to authorize this board operation.")

        with self.lock:
            if self.closed:
                raise ValueError("Chat is shutting down.")

            session = self._load(payload['session_id'])
            if any(job['status'] in ('queued', 'running') for job in self.jobs.values()):
                raise ValueError("An agent request is already running. Wait or stop it first.")

            board = self.app.resolve_board_path(session['board'])
            result_path = None
            if action == 'read':
                result_path = str(self._allowed_result(payload.get('result_path', '')))

            tools = ['VisualInspectionPreflight']
            if action in ('inspect', 'refresh'):
                tools.extend(['InspectBoard', 'ReadInspectionResult'])
            elif action == 'read':
                tools = ['ReadInspectionResult']

            policy = {
                'action': action,
                'tools': tools,
                'config_path': str(self.config.inspection_config),
                'board_path': str(board),
                'result_path': result_path
            }
            if hasattr(self.app, 'config'):
                policy['config_path'] = self.app.config.source_path
                policy['runtime_state'] = {
                    'visual_inspection_repo': str(REPO_ROOT),
                    'visual_inspection_python': sys.executable,
                    'visual_inspection_data_roots': json.dumps([str(REPO_ROOT), str(Path(self.app.config.data_root).resolve())]),
                    'visual_inspection_writable_data_roots': json.dumps([str(Path(self.app.config.data_root).resolve())])
                }

            if action == 'chat':
                snapshot = board.with_suffix('.inspection.json')
                if snapshot.is_file():
                    snapshot = self._allowed_result(str(snapshot))
                    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                    if not any(card.get('sha256') == digest for card in session['results']):
                        entry = {
                            'tool': 'SavedInspection',
                            'envelope': {'result': {
                                'result_path': str(snapshot),
                                'freshness': {'status': 'unverified'}
                            }}
                        }
                        self._capture(session, entry, policy)

            history = copy.deepcopy(session['messages'][-12:])
            history.append({'role': 'evidence', 'results': session['results'][-3:]})
            session['messages'].append({'role': 'user', 'text': message, 'action': action})
            self._save(session)
            job = {
                'id': uuid.uuid4().hex,
                'session_id': session['id'],
                'status': 'queued',
                'action': action,
                'board_path': str(board),
                'command': f"{"Recompute" if action == 'refresh' else "Inspect"} {board}",
                'events': [],
                'cancel': threading.Event(),
                'error': None
            }
            # Keep completed job state bounded; transcripts/evidence remain on disk.
            if len(self.jobs) >= 30:
                self.jobs.pop(next(iter(self.jobs)))

            self.jobs[job['id']] = job
            worker = threading.Thread(target=self._run, args=(job, policy, message, history), daemon=True)
            job['thread'] = worker
            worker.start()
            return self._public(job)

    def _run(self, job: Dict, policy: Dict, message: str, history: List[Dict]) -> None:
        session = None
        agent = None
        try:
            with self.app.execution_guard():
                with self.lock:
                    job['status'] = 'running'

                def event(value: Dict) -> None:
                    item = value['payload'].get('item', {})
                    if item.get('type') == 'mcpToolCall':
                        with self.lock:
                            job['events'].append({'tool': item['tool'], 'status': item['status']})
                            job['events'] = job['events'][-30:]

                agent = ChatSession(
                    replace(self.config, board_path=Path(policy['board_path'])),
                    policy,
                    on_event=event,
                    cancelled=job['cancel']
                )
                with agent:
                    answer = agent.chat(message, history)

                with self.lock:
                    session = self._load(job['session_id'])
                    session['messages'].append({'role': 'assistant', 'text': answer['answer']})
                    for entry in agent.report['evidence']:
                        self._capture(session, entry, policy)

                    self._register_logs(session, agent.run_dir)
                    self._save(session)
                    job['status'] = 'completed'
        except Exception as error:
            with self.lock:
                job['status'] = 'cancelled' if job['cancel'].is_set() else 'failed'
                job['error'] = str(error)
                session = self._load(job['session_id'])
                session['messages'].append({'role': 'error', 'text': str(error)})
                if agent is not None and agent.run_dir.exists():
                    # A tool can finish writing before cancellation or a later model failure.
                    for entry in agent.report.get('evidence', []):
                        try:
                            self._capture(session, entry, policy)
                        except (ValueError, OSError, KeyError):
                            pass

                    self._register_logs(session, agent.run_dir)

                self._save(session)

    def _register_logs(self, session: Dict, directory: Path) -> None:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log') and path.name != 'policy.json':
                try:
                    self._register(session, path)
                except (ValueError, OSError):
                    continue

    def _register(self, session: Dict, path: Path) -> str:
        path = self._allowed_result(str(path))
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        session['artifacts'][identity] = {'path': str(path), 'name': path.name}
        return identity

    def _allowed_result(self, value: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("Select a saved evidence path.")

        path = Path(value).resolve()
        if not any(path.is_relative_to(root) for root in self.result_roots) or not path.is_file():
            raise ValueError("Evidence must be an existing file within configured read/artifact roots.")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Evidence file exceeds the 32 MiB limit.")

        return path

    def _capture(self, session: Dict, entry: Dict, policy: Dict) -> None:
        envelope = entry['envelope']
        payload = envelope['result']
        card = {'tool': entry['tool'], 'payload': payload}
        if entry['tool'] != 'VisualInspectionPreflight':
            path = self._allowed_result(payload['result_path'])
            result = self._validated_snapshot(path, session['board'])
            card['artifact_id'] = self._register(session, path)
            card['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            card['inspection'] = result.to_dict()
            card['freshness'] = payload.get('freshness', {'status': 'validated_at_execution'})
            card['cache_status'] = payload.get('cache_status', 'unknown')
            card['result_id'] = uuid.uuid4().hex
            report = self.config.output_dir / 'reports' / (card['result_id'] + '.txt')
            report.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "Hexaboard inspection evidence",
                f"Board: {session['board']}",
                f"Snapshot: {path}",
                f"SHA256: {card['sha256']}",
                f"Cache: {card['cache_status']}; freshness: {card['freshness']['status']}",
                "Flags are model findings. Final decisions require human review.",
                "Coordinates below are zero-based source (row, column)."
            ]
            for name in ('pw_flags', 'ae_flags', 'hybrid_flags'):
                grid = getattr(result, name)
                flagged = [(row, col) for row in range(8) for col in range(5) if grid[row, col] == 1]
                lines.append(f"{name}: {len(flagged)} flagged: {flagged}")

            report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            card['report_id'] = self._register(session, report)

        if card not in session['results']:
            session['results'].append(card)
            session['results'] = session['results'][-30:]

    def _validated_snapshot(self, path: Path, board: str) -> InspectionResults:
        document = json.loads(path.read_text())
        if document.get('schema_version', 1) != 1 or document.get('metadata', {}).get('board_shape') != [8, 5]:
            raise ValueError("Only whole-segment result snapshots are supported.")
        if Path(document['inspected_path']).resolve() != Path(board).resolve():
            raise ValueError("The saved result belongs to a different board. Start a conversation for that board.")

        result = InspectionResults.load_json(str(path))
        for row in range(8):
            for col in range(5):
                pw, ae, hybrid = (getattr(result, key)[row, col] for key in ('pw_flags', 'ae_flags', 'hybrid_flags'))
                if (pw == -1 or ae == -1 or hybrid == -1) and not pw == ae == hybrid == -1:
                    raise ValueError("Snapshot skip masks disagree.")
                if pw != -1 and hybrid != int(pw == 1 and ae == 1):
                    raise ValueError("Snapshot hybrid flags are inconsistent.")

        return result

    def artifact(self, identity: str, artifact_id: str) -> Path:
        with self.lock:
            session = self._load(identity)
            return self._allowed_result(session['artifacts'][artifact_id]['path'])

    def evidence(self, identity: str, result_id: str) -> Dict:
        with self.lock:
            session = self._load(identity)
            card = next((item for item in session['results'] if item.get('result_id') == result_id), None)
            if card is None:
                raise ValueError("Unknown inspection snapshot.")

            path = self._allowed_result(session['artifacts'][card['artifact_id']]['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != card['sha256']:
                raise ValueError("Saved snapshot changed; load and review it again.")

            inspection = self._validated_snapshot(path, session['board'])
            board = self.app.resolve_board_path(session['board'])
            payload = self.app.build_board_payload(board, inspection=inspection)
            payload['review_context'] = {
                'session_id': identity,
                'result_id': result_id,
                'sha256': card['sha256'],
                'reviews': session['reviews'].get(card['sha256'], {}),
                'freshness': card['freshness']
            }
            return payload

    def review(self, payload: Dict) -> Dict:
        row, col = payload['row'], payload['col']
        if type(row) is not int or type(col) is not int or not 0 <= row < 8 or not 0 <= col < 5:
            raise ValueError("Invalid source segment coordinates.")
        if payload.get('decision') not in ('ok', 'damaged', 'uncertain'):
            raise ValueError("Select a human review decision.")

        reviewer = payload.get('reviewer', '')
        if not isinstance(reviewer, str):
            raise ValueError("Reviewer must be a name.")

        reviewer = reviewer.strip()
        if not reviewer or len(reviewer) > 100:
            raise ValueError("Enter your reviewer name (up to 100 characters).")

        with self.lock:
            evidence = self.evidence(payload['session_id'], payload['result_id'])
            if evidence['inspection']['pw_flags'][row][col] == -1:
                raise ValueError("Skipped segments cannot receive inspection review decisions.")

            session = self._load(payload['session_id'])
            digest = evidence['review_context']['sha256']
            review = session['reviews'].setdefault(digest, {})
            review[f'{row},{col}'] = {
                'decision': payload['decision'],
                'reviewer': reviewer,
                'reviewed_at': datetime.now(timezone.utc).isoformat()
            }
            self._save(session)
            return {'reviews': review}

    def cancel(self, job_id: str) -> Dict:
        with self.lock:
            job = self.jobs[job_id]
            job['cancel'].set()
            return self._public(job)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
            for job in jobs:
                job['cancel'].set()

        for job in jobs:
            job['thread'].join(timeout=8)
