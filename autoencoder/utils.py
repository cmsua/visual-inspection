from torchvision import transforms

# Function to get segments from the hexaboard
def get_segments(image, height: int, width: int, vertical_segments: int, horizontal_segments: int, rotation: int | None = None):
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