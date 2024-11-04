# Import necessary dependencies
from typing import List, Optional
from PIL import Image
from torchvision import transforms

# Function to get segments from an image using 3 rotations
def get_segments(
    image: Image.Image, height: int, width: int,
    vertical_segments: int, horizontal_segments: int,
    rotation: Optional[int] = None
) -> List[Image.Image]:
    """
    Splits an image into smaller segments based on specified vertical and horizontal segments, 
    optionally applying a rotation.

    Args:
        image (Image.Image): The input PIL image to segment.
        height (int): The height of the region to crop from the image.
        width (int): The width of the region to crop from the image.
        vertical_segments (int): Number of vertical segments to split the image into.
        horizontal_segments (int): Number of horizontal segments to split the image into.
        rotation (Optional[int], optional): Rotation angle in degrees to apply to the image.
                                            If None, no rotation is applied. Defaults to None.

    Returns:
        segments (List[Image.Image]): A list of segmented image patches.
    """
    a = width / 2
    
    # Transform and crop image
    transform = None
    if rotation is None: 
        transform = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.CenterCrop((height, int(a)))
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.RandomRotation(degrees=(rotation, rotation)),
            transforms.Resize((height, width)),
            transforms.CenterCrop((height, int(a)))
        ])

    cropped_image = transform(image)

    # Calculate the dimensions of each segment
    segment_height = int(height / vertical_segments)
    segment_width = int(a / horizontal_segments)

    segments = []

    # Split the cropped image into subsegments
    for i in range(vertical_segments):
        for j in range(horizontal_segments):
            left = j * segment_width
            upper = i * segment_height
            right = left + segment_width
            lower = upper + segment_height

            segment = cropped_image.crop((left, upper, right, lower))
            segments.append(segment)

    return segments