from torchvision import transforms

def get_segments_1(image, height, width, vertical_segments, horizontal_segments):
    a = width / 2
    transform = transforms.Compose([
        transforms.CenterCrop((int(height), int(width))),
        transforms.CenterCrop((int(height), int(a)))
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


def get_segments_2(image, height, width, vertical_segments, horizontal_segments):
    a = width / 2
    transform = transforms.Compose([
        transforms.CenterCrop((int(height), int(width))),
        transforms.RandomRotation(degrees=(60, 60)),
        transforms.Resize((height, width)),
        transforms.CenterCrop((int(height), int(a)))
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

def get_segments_3(image, height, width, vertical_segments, horizontal_segments):
    a = width / 2
    transform = transforms.Compose([
        transforms.CenterCrop((int(height), int(width))),
        transforms.RandomRotation(degrees=(240, 240)),
        transforms.Resize((height, width)),
        transforms.CenterCrop((int(height), int(a)))
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