from __future__ import annotations

from http.server import ThreadingHTTPServer

from ..app import InspectionApp
from ..configs import INTERFACE_DIR
from .handlers import InspectionRequestHandler


def run_server(host: str = '127.0.0.1', port: int = 3000) -> None:
    if not INTERFACE_DIR.exists():
        raise RuntimeError(f"Frontend directory not found: {INTERFACE_DIR}")

    server = ThreadingHTTPServer((host, port), InspectionRequestHandler)
    server.app = InspectionApp()
    print(f"Serving inspection UI at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()