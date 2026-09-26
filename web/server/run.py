from __future__ import annotations

from http.server import ThreadingHTTPServer
from typing import Optional

from src.configs.inspection_config import InspectionConfig

from .handlers import InspectionRequestHandler
from ..app import InspectionApp
from ..configs import INTERFACE_DIR


def run_server(host: str = '127.0.0.1', port: int = 3000, config: Optional[InspectionConfig] = None) -> None:
    if not INTERFACE_DIR.exists():
        raise RuntimeError(f"Frontend directory not found: {INTERFACE_DIR}")

    server = ThreadingHTTPServer((host, port), InspectionRequestHandler)
    server.app = InspectionApp(config)
    print(f"Serving inspection UI at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
