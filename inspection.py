# Import necessary dependencies
import argparse

import numpy as np
import cv2

from autoencoder.image_lineup import adjust_image


class ScanResult():
    def __init__(self, base: np.ndarray, annotated: np.ndarray = None) -> None:
        self.base = base
        self.annotated = annotated

def run_inspection(image: np.ndarray) -> ScanResult:
    result = ScanResult(image)
    annotated = image.copy()

    # Do things here

    result.annotated = annotated
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Visual Inspection',
        description='Inspects Hexaboards for defects',
        epilog='University of Alabama'
    )

    # Set up arguments
    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='store_true')

    # Parse
    args = parser.parse_args()

    # Run program
    image = cv2.imload(args.filename)
    result = run_inspection(image)

    cv2.imshow("Inspection Result", result.annotated)