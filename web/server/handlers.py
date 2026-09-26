from __future__ import annotations

import json
import subprocess
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from ..app import InspectionApp
from ..configs import INTERFACE_DIR
from ..utils import repo_relative


class InspectionRequestHandler(SimpleHTTPRequestHandler):
    server_version = 'HexaInspection/0.1'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(INTERFACE_DIR), **kwargs)

    @property
    def app(self) -> InspectionApp:
        return self.server.app  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith('/api/'):
            try:
                self.handle_api_get(parsed.path, parse_qs(parsed.query))
            except FileNotFoundError:
                self._send_json(
                    {'error': "Required board or inspection artifact not found"},
                    status=HTTPStatus.NOT_FOUND
                )
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json({'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pylint: disable=broad-except
                self.log_error("GET %s failed: %s", parsed.path, exc)
                self._send_json({'error': "Internal server error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            return

        super().do_GET()

    def handle_api_get(self, path: str, params: Dict[str, List[str]]) -> None:
        if path == '/api/config':
            self._send_json({'board_path': repo_relative(Path(self.app.config.board_path))})
            return

        if path == '/api/board':
            board_param = params.get('path', [self.app.config.board_path])[0]
            board_path = self._resolve_board_path(board_param)
            payload = self.app.build_board_payload(board_path)
            self._send_json(payload)

            return

        if path == '/api/segment':
            board_param = params.get('path', [self.app.config.board_path])[0]
            row = int(params.get('row', [0])[0])
            col = int(params.get('col', [0])[0])
            board_path = self._resolve_board_path(board_param)
            payload = self.app.segment_image_payload(board_path, row=row, col=col)
            self._send_json(payload)

            return

        self._send_json({'error': f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith('/api/'):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 65536:
                    raise ValueError("Request body is too large.")

                body = self.rfile.read(length) if length > 0 else b'{}'
                payload = json.loads(body.decode('utf-8')) if body else {}
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")

                self.handle_api_post(parsed.path, payload)
            except json.JSONDecodeError:
                self._send_json({'error': "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            except FileNotFoundError:
                self._send_json(
                    {'error': "Required board or inspection artifact not found"},
                    status=HTTPStatus.NOT_FOUND
                )
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json({'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except subprocess.CalledProcessError as exc:
                self.log_error("Process failed: %s", exc)
                self._send_json({'error': "Inspection run failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception as exc:  # pylint: disable=broad-except
                self.log_error("POST %s failed: %s", parsed.path, exc)
                self._send_json({'error': "Internal server error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_api_post(self, path: str, payload: Dict[str, object]) -> None:
        if path == '/api/label':
            board_param = str(payload.get('path') or self.app.config.board_path)
            if type(payload['row']) is not int or type(payload['col']) is not int:
                raise ValueError("Label coordinates must be integers.")

            row = int(payload['row'])
            col = int(payload['col'])
            is_damaged = payload.get('damaged', False)
            if not isinstance(is_damaged, bool):
                raise ValueError("`damaged` must be a boolean.")

            board_path = self._resolve_board_path(board_param)
            array = self.app._load_board(board_path)  # pylint: disable=protected-access
            self.app.update_damaged_label(board_path, array.shape, row=row, col=col, is_damaged=is_damaged)
            updated = self.app.get_damaged_labels(board_path, array.shape)
            self._send_json({'damaged_segments': updated})

            return

        if path == '/api/run':
            board_param = str(payload.get('path') or self.app.config.board_path)
            board_path = self._resolve_board_path(board_param)
            result = self.app.run_inspection_job(board_path)
            status = 'ok' if result.returncode == 0 else 'error'
            self._send_json(
                {'status': status, 'stdout': result.stdout, 'stderr': result.stderr},
                status=HTTPStatus.OK if result.returncode == 0 else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            return

        self._send_json({'error': f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _resolve_board_path(self, board_param: str) -> Path:
        return self.app.resolve_board_path(board_param)

    def _send_json(self, payload: Dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', "application/json; charset=utf-8")
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
