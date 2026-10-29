import argparse
from pathlib import Path

from src.configs import DEFAULT_INSPECTION_CONFIG, InspectionConfig
from web.configs import AgentConfig
from web.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the hexaboard inspection interface.")
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--agent-config', help="Enable Agent chat using an agent YAML.")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=3000)
    args = parser.parse_args()
    agent = AgentConfig.from_yaml(Path(args.agent_config)) if args.agent_config else None
    config_path = args.config_path or str(DEFAULT_INSPECTION_CONFIG)
    config = InspectionConfig.from_yaml(config_path)
    run_server(args.host, args.port, config, agent_config=agent)


if __name__ == '__main__':
    main()
