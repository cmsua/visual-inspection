import os

from torchvision.transforms import ToTensor
import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as ssim

from model import SimpleCNNAutoEncoder
from inspection.calibrate_metrics import calibrate_metrics

def ae_inference(device, CHECKPOINT_PATH, newSegments, baselineSegments):
    
    segment_width, segment_height = newSegments[0].size
    
    model = SimpleCNNAutoEncoder(
        height=segment_height,
        width=segment_width,
        latent_dim=128,
        kernel_sizes=[64, 128]
    )
    model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # Evaluate the segments
    flaggedML = []
    threshold = 0.4

    for i, (seg1, seg2) in enumerate(zip(newSegments, baselineSegments)):

        # output should be compared to seg1, not seg2?
        seg1 = ToTensor()(seg1).unsqueeze(0)
        seg2 = np.array(seg2)

        with torch.no_grad():
            output = model(seg1).cpu().squeeze().permute(1, 2, 0).numpy()
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            seg2 = cv2.cvtColor(seg2, cv2.COLOR_BGR2GRAY)

            # SSIM metric
            ssim_val, _ = ssim(output, seg2, full=True, data_range=output.max() - output.min())

            if ssim_val < threshold:
                flaggedML.append(i)

    ae_optimal_threshold, bad_values, good_values = calibrate_metrics(output, seg1, flaggedML)
    ae_values = np.concatenate(bad_values, good_values)

    return flaggedML, ae_optimal_threshold, ae_values