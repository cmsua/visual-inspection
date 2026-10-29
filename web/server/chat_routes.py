from http import HTTPStatus
from typing import Dict, List
from urllib.parse import urlparse


def validate_origin(handler: object) -> None:
    host = handler.headers.get('Host', '')
    if getattr(handler.server, 'chat', None) is not None and urlparse('//' + host).hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError("Agent requests must use a loopback host name.")

    origin = handler.headers.get('Origin')
    if handler.headers.get('Sec-Fetch-Site') == 'cross-site':
        raise ValueError("Cross-site requests are not allowed.")
    if origin and urlparse(origin).netloc != host:
        raise ValueError("The request origin must match this local server.")


def chat_get(handler: object, path: str, params: Dict[str, List[str]]) -> bool:
    if not path.startswith('/api/chat/'):
        return False

    app = getattr(handler.server, 'chat', None)
    if app is None:
        handler._send_json({'enabled': False, 'error': "Start the server with --agent-config to enable chat."})
        return True

    def value(name: str) -> str:
        return params[name][0]

    if path == '/api/chat/status':
        handler._send_json(app.status())
    elif path == '/api/chat/session':
        handler._send_json(app.get(value('id')))
    elif path == '/api/chat/evidence':
        handler._send_json(app.evidence(value('session_id'), value('result_id')))
    elif path == '/api/chat/artifact':
        artifact = app.artifact(value('session_id'), value('id'))
        data = artifact.read_bytes()
        handler.send_response(HTTPStatus.OK)
        handler.send_header('Content-Type', 'text/plain; charset=utf-8')
        handler.send_header('Content-Disposition', 'inline; filename="inspection-evidence.txt"')
        handler.send_header('X-Content-Type-Options', 'nosniff')
        handler.send_header('Content-Length', str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    else:
        handler._send_json({'error': "Unknown chat endpoint"}, status=HTTPStatus.NOT_FOUND)

    return True


def chat_post(handler: object, path: str, payload: Dict) -> bool:
    if not path.startswith('/api/chat/'):
        return False

    app = getattr(handler.server, 'chat', None)
    if app is None:
        raise ValueError("Agent chat is not enabled.")

    routes = {
        '/api/chat/request': lambda: app.request(payload),
        '/api/chat/sessions': lambda: app.create(payload['board_path']),
        '/api/chat/messages': lambda: app.submit(payload),
        '/api/chat/cancel': lambda: app.cancel(payload['job_id']),
        '/api/chat/review': lambda: app.review(payload)
    }
    if path not in routes:
        raise ValueError("Unknown chat endpoint.")

    handler._send_json(routes[path]())
    return True
