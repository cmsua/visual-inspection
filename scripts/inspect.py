import argparse
import json

from src.configs.inspection_config import DEFAULT_INSPECTION_CONFIG, InspectionConfig
from src.inspection import inspect_board


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a processed board and persist its result JSON.")
    parser.add_argument('--config-path', default=str(DEFAULT_INSPECTION_CONFIG))
    parser.add_argument('-n', '--new-hexaboard-path', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    results = inspect_board(InspectionConfig.from_yaml(args.config_path), args.new_hexaboard_path, force=args.force)
    print(json.dumps(results.summary(), indent=4))


if __name__ == '__main__':
    main()
