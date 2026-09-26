import argparse

from src.configs.inspection_config import DEFAULT_INSPECTION_CONFIG, InspectionConfig
from web.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the hexaboard inspection interface.")
    parser.add_argument('--config-path', default=str(DEFAULT_INSPECTION_CONFIG))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=3000)
    args = parser.parse_args()
    run_server(args.host, args.port, InspectionConfig.from_yaml(args.config_path))


if __name__ == '__main__':
    main()
