import argparse

from src.configs.inspection_config import DEFAULT_INSPECTION_CONFIG, InspectionConfig
from src.inspection.calibration import calibrate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pixel and AE thresholds for the configured checkpoint.")
    parser.add_argument('--config-path', default=str(DEFAULT_INSPECTION_CONFIG))
    args = parser.parse_args()
    calibrate(InspectionConfig.from_yaml(args.config_path))


if __name__ == '__main__':
    main()
