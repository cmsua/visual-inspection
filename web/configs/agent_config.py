import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


PREFLIGHT_TOOL = 'VisualInspectionPreflight'
SERVER_NAME = 'heptapod_preflight'


@dataclass(frozen=True)
class AgentConfig:
    heptapod_repo: Path
    heptapod_python: Path
    state_config: Path
    inspection_config: Path
    board_path: Optional[Path]
    output_dir: Path
    codex: str = 'codex'
    model: Optional[str] = None
    timeout_seconds: float = 180

    @classmethod
    def from_yaml(cls, path: Path) -> 'AgentConfig':
        path = path.resolve()
        values = yaml.safe_load(path.read_text(encoding='utf-8'))['agent_preflight']
        root = path.parent

        def absolute(value: str) -> Path:
            # Do not resolve the interpreter symlink: it identifies the venv.
            candidate = Path(value).expanduser()
            return candidate if candidate.is_absolute() else root / candidate

        timeout = float(values.get('timeout_seconds', 180))
        if not math.isfinite(timeout) or timeout <= 0 or timeout > 1800:
            raise ValueError("Agent timeout must be between zero and 1800 seconds.")

        return cls(
            heptapod_repo=absolute(values['heptapod_repo']).resolve(),
            heptapod_python=absolute(values['heptapod_python']),
            state_config=absolute(values['state_config']).resolve(),
            inspection_config=absolute(values['inspection_config']).resolve(),
            board_path=absolute(values['board_path']).resolve() if values.get('board_path') else None,
            output_dir=absolute(values['output_dir']).resolve(),
            codex=values.get('codex', 'codex'),
            model=values.get('model'),
            timeout_seconds=timeout
        )

    def server_config(self, run_dir: Path, tools: Tuple[str, ...] = (PREFLIGHT_TOOL,)) -> Dict:
        for path in (self.heptapod_python, self.state_config, self.inspection_config):
            if not path.is_file():
                raise ValueError(f"Required file does not exist: `{path}`.")

        metadata = yaml.safe_load((self.heptapod_repo / 'toolkit.yaml').read_text(encoding='utf-8'))
        entries = [
            entry for entry in metadata['tools']
            if entry.get('bundle') == 'visual_inspection' and entry['name'] in {name + 'Tool' for name in tools}
        ]
        if len(entries) != len(tools):
            raise ValueError("HEPTAPOD does not register all requested inspection tools.")

        state = yaml.safe_load(self.state_config.read_text(encoding='utf-8'))
        state = {key: value for key, value in state.items() if key.startswith('visual_inspection_')}
        required = ('visual_inspection_repo', 'visual_inspection_python', 'visual_inspection_data_roots')
        if any(not state.get(key) for key in required):
            raise ValueError("HEPTAPOD state is missing repository, interpreter, or read-root settings.")

        state['visual_inspection_artifact_root'] = str(run_dir / 'heptapod')
        state['visual_inspection_timeout'] = min(float(state.get('visual_inspection_timeout', 60)), self.timeout_seconds)
        return {
            'command': str(self.heptapod_python),
            'args': [
                '-B', '-m', 'toolbase._toolkit_host',
                '--toolkit-dir', str(self.heptapod_repo),
                '--name', 'heptapod',
                '--state-config', json.dumps(state),
                '--tools-spec', json.dumps(entries)
            ],
            'cwd': str(self.heptapod_repo),
            'env': {
                'PYTHONDONTWRITEBYTECODE': '1',
                'TOOLBASE_HOST_LOG': str(run_dir / 'toolbase.log')
            },
            'enabled_tools': list(tools),
            'tools': {name: {'approval_mode': 'approve'} for name in tools},
            'required': True,
            'startup_timeout_sec': 45,
            'tool_timeout_sec': self.timeout_seconds
        }
