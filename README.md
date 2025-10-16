# Uncertainty-Guided Progressive U-Net

This repository contains an implementation of an Uncertainty-Guided Progressive U-Net for medical image analysis. We tested it on 2 tasks: nuclei segmentation (MoNuSeg) and cervical cell classification (Herlev).

## 🏗️ Architecture

The Uncertainty-Guided Progressive U-Net is a multi-stage architecture that progressively increases resolution from 32×32 to 256×256 pixels:

- **Stage 1**: 32×32 resolution - Basic feature learning
- **Stage 2**: 64×64 resolution - Incorporates uncertainty from Stage 1
- **Stage 3**: 128×128 resolution - Incorporates uncertainties from Stages 1-2
- **Stage 4**: 256×256 resolution - Final high-resolution prediction

### Key Features

- **Progressive Training**: Each stage is trained sequentially with weight transfer
- **Uncertainty Guidance**: Later stages use uncertainty maps from earlier stages
- **Modular Design**: Support for both segmentation and classification tasks
- **Stage Flexibility**: Can train individual stages or full progressive pipeline

## 📁 Project Structure

```
UncertainGuidePGU/
├── UG_unet.py                      # Core model implementations
├── UG_unet_parts.py                # U-Net building blocks
├── uncertainty_guided_trainer.py    # Progressive training framework
├── demo_uncertainty_guided.py      # Usage examples
```

Install core requirements:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Model Usage

```python
from UG_unet import ProgressiveUNet, UncertaintyGuidedLoss
from uncertainty_guided_trainer import UncertaintyGuidedProgressiveTrainer

# Create model for segmentation (default)
model = ProgressiveUNet(in_channels=3, out_channels=2, stage=1)

# Create model for classification
model = ProgressiveUNet(in_channels=3, out_channels=7, stage=4, task_type='classification')

# Create uncertainty-guided loss
criterion = UncertaintyGuidedLoss(task_type='segmentation')
```

## Our training samples

### 2. MoNuSeg Nuclei Segmentation

#### Training

```bash
cd MoNuSegImprove
python train_aug_monuseg.py --data_dir ./train --output_dir ./outputs --epochs 100
```

#### Evaluation

```bash
python test_monuseg.py --model ./outputs/pgunet_stage4_best.pth --data ./val --split test
```

### 3. Herlev Cervical Cell Classification

#### Training

```bash
cd Herlev
python train_herlev.py --data_dir "./data/Herlev Dataset/train" --output_dir ./outputs --epochs 50
```

#### Evaluation

```bash
python test_herlev.py --model ./outputs/herlev_stage4_best.pth --data "./data/Herlev Dataset/train" --split test
```

## 📊 Datasets

### MoNuSeg Dataset

The Multi-organ Nuclei Segmentation (MoNuSeg) dataset for nuclei segmentation:

- **Task**: Binary segmentation (background vs nuclei)
- **Format**: Images with XML annotations
- **Classes**: 2 (background, nuclei)

Expected structure:

```
MoNuSegImprove/
├── train/
│   ├── images/
│   └── annots/
└── val/
    ├── images/
    └── annots/
```

### Herlev Dataset

The Herlev cervical cell dataset for cytology classification:

- **Task**: 7-class cervical cell classification
- **Classes**:
  - normal_columnar
  - normal_intermediate
  - normal_superficiel
  - light_dysplastic
  - moderate_dysplastic
  - severe_dysplastic
  - carcinoma_in_situ

Expected structure:

```
Herlev/data/Herlev Dataset/train/
├── normal_columnar/
├── normal_intermediate/
├── normal_superficiel/
├── light_dysplastic/
├── moderate_dysplastic/
├── severe_dysplastic/
└── carcinoma_in_situ/
```

## 🎯 Training Process

### Progressive Training Stages

1. **Stage 1 (32×32)**: Train base model
2. **Stage 2 (64×64)**: Transfer Stage 1 weights + uncertainty guidance
3. **Stage 3 (128×128)**: Transfer Stage 2 weights + multi-stage uncertainty
4. **Stage 4 (256×256)**: Transfer Stage 3 weights + full uncertainty guidance

### Training Configuration

```python
config = {
    'batch_size': 16,
    'learning_rate': 1e-3,
    'epochs_per_stage': 50,
    'uncertainty_weight': 0.1,
    'early_stopping_patience': 10
}
```

## 📈 Model Performance

### Evaluation Metrics

- **Segmentation**: Dice Score, IoU, Pixel Accuracy
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Output Files

- Model checkpoints: `pgunet_stage{1-4}_best.pth`
- Training logs: `training_log.csv`
- Evaluation results: `evaluation_results.json`
- Visualizations: Confusion matrices, sample predictions

## 🔍 Advanced Usage

### Custom Dataset Integration

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, image_size=256, transform=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        # Implement dataset loading logic

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return (image, label) tuple
        pass
```

### Custom Training Loop

```python
from uncertainty_guided_trainer import UncertaintyGuidedProgressiveTrainer

class CustomTrainer(UncertaintyGuidedProgressiveTrainer):
    def setup_datasets(self):
        # Implement custom dataset setup
        pass

    def validate_epoch(self, stage):
        # Implement custom validation logic
        pass
```

## 🙏 Acknowledgments

- Original U-Net architecture by Ronneberger et al.
- Progressive U-Net concepts inspired by [Springer Chapter](https://link.springer.com/chapter/10.1007/978-3-030-37969-8_7)
- MoNuSeg dataset providers
- Herlev dataset contributors
- PyTorch community for excellent deep learning framework
