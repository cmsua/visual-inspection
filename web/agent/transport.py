import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


class AgentError(RuntimeError):
    pass


def _process_info(pid: int) -> Optional[List[str]]:
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (OSError, IndexError):
        return None


def _descendants(pid: int) -> Dict[int, str]:
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = _process_info(int(path.name))
            if info is not None:
                processes[int(path.name)] = info

    descendants = {pid: ''}
    changed = True
    while changed:
        changed = False
        for child, info in processes.items():
            if child not in descendants and int(info[1]) in descendants:
                descendants[child] = info[19]
                changed = True

    descendants.pop(pid)
    return descendants


def _signal_descendants(descendants: Dict[int, str], sig: int) -> None:
    for pid, started in descendants.items():
        info = _process_info(pid)
        if info is not None and info[19] == started:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


class CodexTransport:
    def __init__(self, command: List[str], cwd: Path, stderr_path: Path) -> None:
        self.command = command
        self.cwd = cwd
        self.stderr_path = stderr_path
        self.process = None
        self.log = None
        self.messages = queue.Queue(maxsize=1024)
        self.stopping = threading.Event()
        self.counter = 0

    def __enter__(self) -> 'CodexTransport':
        self.log = self.stderr_path.open('xb')
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                start_new_session=True
            )
        except OSError:
            self.log.close()
            raise

        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        return self

    def _enqueue(self, item: object) -> None:
        while not self.stopping.is_set():
            try:
                self.messages.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _read(self) -> None:
        try:
            while not self.stopping.is_set():
                line = self.process.stdout.readline(4 * 1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 4 * 1024 * 1024:
                    raise AgentError("Codex sent an oversized protocol message.")

                message = json.loads(line)
                if not isinstance(message, dict):
                    raise AgentError("Codex sent an invalid protocol message.")

                self._enqueue(message)
        except (ValueError, OSError, AgentError) as error:
            self._enqueue(AgentError(f"Invalid Codex output: {error}"))
        finally:
            self._enqueue(AgentError("Codex disconnected; inspect codex.stderr.log."))

    def send(self, message: Dict) -> None:
        try:
            self.process.stdin.write((json.dumps(message, allow_nan=False) + '\n').encode('utf-8'))
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise AgentError("Unable to send a request to Codex.") from error

    def request(self, method: str, params: Dict) -> int:
        self.counter += 1
        self.send({'id': self.counter, 'method': method, 'params': params})
        return self.counter

    def receive(self, deadline: float, cancelled: Optional[threading.Event] = None) -> Dict:
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled.is_set():
                raise AgentError("Conversation cancelled. Diagnostic artifacts may already exist.")

            try:
                message = self.messages.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue

            if isinstance(message, Exception):
                raise message

            return message

        raise AgentError("Codex timed out. Inspect retained diagnostics before retrying.")

    def __exit__(self, *args: object) -> None:
        self.stopping.set()
        if self.process is not None:
            # HEPTAPOD workers start their own sessions, outside Codex's group.
            descendants = _descendants(self.process.pid)
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            _signal_descendants(descendants, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            _signal_descendants(descendants, signal.SIGKILL)
            self.process.wait(timeout=3)
            self.process.stdin.close()
            self.process.stdout.close()
            self.reader.join(timeout=1)

        if self.log is not None:
            self.log.close()
