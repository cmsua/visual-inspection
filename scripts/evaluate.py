import argparse
import json

from src.configs.inspection_config import DEFAULT_INSPECTION_CONFIG, InspectionConfig
from src.inspection.evaluation import evaluate_reconstruction


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction loss and save an example plot.")
    parser.add_argument('--config-path', default=str(DEFAULT_INSPECTION_CONFIG))
    parser.add_argument('--display-segment-idx', type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_reconstruction(InspectionConfig.from_yaml(args.config_path), args.display_segment_idx),
            indent=4
        )
    )


if __name__ == '__main__':
    main()
