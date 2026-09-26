import argparse

from src.configs.inspection_config import DEFAULT_INSPECTION_CONFIG, InspectionConfig
from src.inspection.analysis import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze held-out boards using calibrated thresholds.")
    parser.add_argument('--config-path', default=str(DEFAULT_INSPECTION_CONFIG))
    parser.add_argument('--max-boards', type=int, default=None)
    args = parser.parse_args()
    if args.max_boards is not None and args.max_boards < 1:
        parser.error("--max-boards must be positive.")

    analyze(InspectionConfig.from_yaml(args.config_path), args.max_boards)


if __name__ == '__main__':
    main()
