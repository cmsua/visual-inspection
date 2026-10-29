from __future__ import annotations

from http.server import ThreadingHTTPServer
from typing import Optional

from src.configs import InspectionConfig

from .handlers import InspectionRequestHandler
from ..app import ChatApp, InspectionApp
from ..configs import INTERFACE_DIR, AgentConfig


def run_server(
    host: str = '127.0.0.1',
    port: int = 3000,
    config: Optional[InspectionConfig] = None,
    agent_config: Optional[AgentConfig] = None
) -> None:
    if not INTERFACE_DIR.exists():
        raise RuntimeError(f"Frontend directory not found: {INTERFACE_DIR}")

    if agent_config is not None and host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError("The agent demo supports loopback hosting only.")

    server = ThreadingHTTPServer((host, port), InspectionRequestHandler)
    server.app = InspectionApp(
        config,
        labels_path=str(agent_config.output_dir / 'manual_labels.json') if agent_config else None
    )
    server.chat = ChatApp(server.app, agent_config) if agent_config else None
    print(f"Serving inspection UI at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if server.chat is not None:
            server.chat.close()

        server.server_close()
