# UA CMS: Hexaboard Visual Inspection

A deep learning-based visual inspection system for detecting defects in hexaboard segments from the High Granularity Calorimeter (HGCAL) detector of the CMS experiment at CERN.

## Description

The High Granularity Calorimeter (HGCAL) is a key component of the CMS detector upgrade for the High Luminosity Large Hadron Collider (HL-LHC). HGCAL consists of silicon sensors arranged in hexagonal modules called hexaboards, which provide unprecedented spatial resolution for particle detection in the forward region of the CMS detector.

![Cutaway diagram of CMS detector (retrieved from https://cds.cern.ch/record/2665537/files/)](assets/cms_160312_02.png)
*Cutaway diagram of CMS detector (retrieved from https://cds.cern.ch/record/2665537/files/)*

Hexaboards are critical silicon sensor modules that form the active detection layers of the HGCAL endcap calorimeter. These hexagonal-shaped boards contain arrays of silicon pad sensors that measure the energy deposits from electromagnetic and hadronic showers. Each hexaboard must meet strict quality standards, as defects can significantly impact the detector's performance in measuring particle energies and positions with high precision.

This project implements an automated visual inspection system that combines:
- **Autoencoder-based anomaly detection**: A ResNet-inspired convolutional autoencoder trained on reference images to detect reconstruction anomalies
- **Pixel-wise comparison**: Traditional image comparison using Structural Similarity Index Measure (SSIM) between baseline and test images
- **Dual flagging system**: Segments are classified as defective if flagged by either or both methods, providing comprehensive defect detection

## Get Started

### Prerequisites

- Python 3.13 or higher
- Git

### Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd visual-inspection
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv

    # On Windows
    venv\Scripts\activate

    # On macOS/Linux
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install PyTorch (GPU or CPU):**

    See: https://pytorch.org/get-started/locally/

## Train/Evaluate the Autoencoder

### Model Architecture

The system uses a ResNet-inspired convolutional autoencoder (`ResNetAutoencoder`) with the following key features:

- **Encoder**: Based on ResNet architecture with BasicBlock modules, progressively downsampling the input
- **Bottleneck**: Compressed latent representation (default: 128 dimensions)
- **Decoder**: Symmetric upsampling path using ConvTranspose2d layers to reconstruct the original image
- **Loss Function**: `BCEWithLogitsLoss` for pixel-wise reconstruction

The model is designed to learn the normal appearance of hexaboard segments. During inference, segments with poor reconstruction quality (low SSIM scores) are flagged as potentially defective.

### Training the Model

To train the autoencoder on your hexaboard data:

```bash
python -m scripts.train \
    --data-path ./data/ref_image_array.npy \
    --latent-dim 128 \
    --batch-size 4 \
    --num-epochs 100 \
    --learning-rate 1e-3 \
    --device cuda
```

**Key training arguments:**
- `--data-path`: Path to the reference hexaboard image array (.npy file)
- `--latent-dim`: Bottleneck dimension (default: 128)
- `--init-filters`: Initial number of filters (default: 64)
- `--layers`: ResNet layer configuration (default: [2, 2, 2])
- `--batch-size`: Training batch size (default: 4)
- `--num-epochs`: Number of training epochs (default: 100)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--early-stopping-patience`: Early stopping patience (default: 10)
- `--log-dir`: Directory to save model checkpoints (default: ./logs)

### Evaluating the Model

To evaluate the trained model and visualize reconstructions:

```bash
python -m scripts.evaluate \
    --data-path ./data/ref_image_array.npy \
    --best-model-path ./logs/ResNetAutoencoder/best/run_01.pt \
    --batch-size 4 \
    --num-images 8
```

**Key evaluation arguments:**
- `--best-model-path`: Path to the trained model weights
- `--num-images`: Number of reconstruction examples to visualize (default: 8)
- `--no-plot`: Disable plotting of reconstructed images

## Run the Inspection

### Basic Usage

To perform visual inspection on hexaboard images:

```bash
python -m src.inspection.main \
    --baseline-images-path ./data/baseline_hexaboard.npy \
    --new-images-path ./data/test_hexaboard.npy \
    --best-model-path ./logs/ResNetAutoencoder/best/run_01.pt \
    --latent-dim 128 \
    --batch-size 4
```

### Command-line Arguments

**Required Arguments:**
- `-b, --baseline-images-path`: Path to baseline hexaboard images (.npy file)
- `-n, --new-images-path`: Path to new hexaboard images to inspect (.npy file)

**Optional Arguments:**
- `-w, --best-model-path`: Path to trained model weights (default: ./logs/ResNetAutoencoder/best/run_01.pt)
- `--latent-dim`: Autoencoder latent dimension (default: 128)
- `--init-filters`: Initial filter count (default: 64)
- `--layers`: ResNet layer configuration (default: [2, 2, 2])
- `--batch-size`: Inference batch size (default: 4)
- `--device`: Computation device (default: auto-detect CUDA/CPU)

### Expected Input Format

The input `.npy` files should contain hexaboard image arrays with shape:
```
(H_seg, V_seg, height, width, num_channels)
```

Where:
- `H_seg`: Number of horizontal segments (typically 12)
- `V_seg`: Number of vertical segments (typically 9)
- `height, width`: Pixel dimensions of each segment
- `num_channels`: Color channels (3 for RGB)

### Output

The inspection system outputs four categories of flagged segments:

1. **Double flagged**: Segments flagged by both autoencoder and pixel-wise methods
2. **Autoencoder flagged**: Segments flagged only by the ML model
3. **Pixel-wise flagged**: Segments flagged only by traditional comparison
4. **All flagged**: Union of all flagged segments

Each flagged segment is identified by its `(board_index, h_segment, v_segment)` coordinates.

### Example Output

```
Double flagged segments: [(0, 3, 2), (0, 7, 5)]
Autoencoder flagged segments: [(0, 1, 8), (0, 9, 3)]
Pixel-wise flagged segments: [(0, 5, 1)]
All flagged segments: [(0, 1, 8), (0, 3, 2), (0, 5, 1), (0, 7, 5), (0, 9, 3)]
```

## Testing

Run the test suite to verify the installation:

```bash
python -m pytest tests/ -v
```

## Project Structure

```
visual-inspection/
├── src/
│   ├── configs/           # Configuration modules
│   ├── engine/            # Training engine
│   ├── inferences/        # Inference implementations
│   ├── inspection/        # Main inspection module
│   ├── models/            # Neural network models
│   └── utils/             # Utility functions and datasets
├── scripts/               # Training and evaluation scripts
├── tests/                 # Unit tests
├── logs/                  # Model checkpoints and logs
├── data/                  # Data directory (add your .npy files here)
└── notebooks/             # Jupyter notebooks for analysis
```