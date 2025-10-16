"""
Herlev Dataset Loader for Cervical Cell Classification

This module provides a PyTorch Dataset class for loading the Herlev Pap Smear dataset
for cervical cell classification tasks. The dataset contains 7 classes of cervical cells
with varying image sizes.

Classes:
- 0: carcinoma_in_situ
- 1: light_dysplastic
- 2: moderate_dysplastic  
- 3: normal_columnar
- 4: normal_intermediate
- 5: normal_superficiel
- 6: severe_dysplastic
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List, Dict, Any, Optional, Union
import random
import json
from pathlib import Path


class HerlevDataset(Dataset):
    """
    Herlev Pap Smear dataset loader for cervical cell classification.
    
    The dataset contains:
    - RGB BMP images with variable sizes
    - 7 classes of cervical cells (normal and abnormal)
    - Class-based directory structure
    """
    
    # Class mappings
    CLASS_NAMES = [
        'carcinoma_in_situ',
        'light_dysplastic', 
        'moderate_dysplastic',
        'normal_columnar',
        'normal_intermediate',
        'normal_superficiel',
        'severe_dysplastic'
    ]
    
    # Group classes into normal vs abnormal for binary classification
    BINARY_MAPPING = {
        'carcinoma_in_situ': 1,      # abnormal
        'light_dysplastic': 1,        # abnormal
        'moderate_dysplastic': 1,     # abnormal
        'normal_columnar': 0,         # normal
        'normal_intermediate': 0,     # normal
        'normal_superficiel': 0,      # normal
        'severe_dysplastic': 1        # abnormal
    }
    
    def __init__(
        self,
        data_dir: str,
        image_size: Union[int, Tuple[int, int]] = 224,
        split: str = 'train',
        transform: bool = True,
        augment: bool = True,
        binary_classification: bool = False,
        normalize: bool = True,
        target_split_ratio: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            data_dir: Path to Herlev dataset directory (should contain train/test folders)
            image_size: Target image size (int for square, tuple for (height, width))
            split: Dataset split ('train', 'val', 'test')
            transform: Whether to apply transforms
            augment: Whether to apply data augmentation (only for train split)
            binary_classification: If True, use binary labels (normal=0, abnormal=1)
            normalize: Whether to normalize images to [0,1] range
            target_split_ratio: Dict with split ratios if creating splits from scratch
                                e.g., {'train': 0.7, 'val': 0.2, 'test': 0.1}
        """
        self.data_dir = data_dir
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.split = split
        self.transform = transform
        self.augment = augment and (split == 'train')  # Only augment training data
        self.binary_classification = binary_classification
        self.normalize = normalize
        
        # Setup class mappings
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.CLASS_NAMES)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load dataset
        self.samples = []
        self.labels = []
        self._load_dataset(target_split_ratio)
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"Herlev {split} dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        if binary_classification:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"  Class distribution: Normal={counts[0] if 0 in unique_labels else 0}, "
                  f"Abnormal={counts[1] if 1 in unique_labels else 0}")
        else:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"    {self.idx_to_class[label]}: {count}")
    
    def _load_dataset(self, target_split_ratio: Optional[Dict[str, float]] = None):
        """Load dataset samples and labels"""
        
        # Try to find existing split structure first
        split_dir = os.path.join(self.data_dir, self.split)
        if os.path.exists(split_dir):
            self._load_from_split_directory(split_dir)
        else:
            # Look for class-based structure and create splits
            train_dir = os.path.join(self.data_dir, 'train')
            if os.path.exists(train_dir) and any(os.path.isdir(os.path.join(train_dir, d)) 
                                                for d in os.listdir(train_dir)):
                self._load_from_class_structure(train_dir, target_split_ratio)
            else:
                # Try alternative structure
                if any(cls in os.listdir(self.data_dir) for cls in self.CLASS_NAMES):
                    self._load_from_class_structure(self.data_dir, target_split_ratio)
                else:
                    raise RuntimeError(f"Cannot find valid Herlev dataset structure in {self.data_dir}")
    
    def _load_from_split_directory(self, split_dir: str):
        """Load from pre-split directory structure"""
        for class_name in self.CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.bmp')]
                for img_file in image_files:
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append(img_path)
                    
                    # Assign label based on classification mode
                    if self.binary_classification:
                        label = self.BINARY_MAPPING[class_name]
                    else:
                        label = self.class_to_idx[class_name]
                    self.labels.append(label)
    
    def _load_from_class_structure(self, base_dir: str, target_split_ratio: Optional[Dict[str, float]] = None):
        """Load from class-based directory structure and create splits"""
        
        # Default split ratios
        if target_split_ratio is None:
            target_split_ratio = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        
        # Collect all samples per class
        all_samples = {}
        for class_name in self.CLASS_NAMES:
            class_dir = os.path.join(base_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.bmp')]
                all_samples[class_name] = [os.path.join(class_dir, f) for f in image_files]
        
        # Create stratified splits
        random.seed(42)  # For reproducible splits
        
        for class_name, class_samples in all_samples.items():
            random.shuffle(class_samples)
            
            n_total = len(class_samples)
            n_train = int(n_total * target_split_ratio['train'])
            n_val = int(n_total * target_split_ratio['val'])
            
            # Split samples
            if self.split == 'train':
                selected_samples = class_samples[:n_train]
            elif self.split == 'val':
                selected_samples = class_samples[n_train:n_train + n_val]
            elif self.split == 'test':
                selected_samples = class_samples[n_train + n_val:]
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            # Add to dataset
            for img_path in selected_samples:
                self.samples.append(img_path)
                
                # Assign label based on classification mode
                if self.binary_classification:
                    label = self.BINARY_MAPPING[class_name]
                else:
                    label = self.class_to_idx[class_name]
                self.labels.append(label)
    
    def _setup_transforms(self):
        """Setup image transforms"""
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize(self.image_size))
        
        # Data augmentation for training
        if self.augment and self.split == 'train':
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.normalize:
            # Use ImageNet normalization as default
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        self.transforms = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', self.image_size, color=(0, 0, 0))
        
        # Apply transforms
        if self.transform and self.transforms:
            image = self.transforms(image)
        else:
            # Basic tensor conversion
            image = TF.to_tensor(image)
            image = TF.resize(image, self.image_size)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            Tensor of class weights (inverse frequency weighting)
        """
        if self.binary_classification:
            n_classes = 2
        else:
            n_classes = len(self.CLASS_NAMES)
        
        class_counts = np.bincount(self.labels, minlength=n_classes)
        total_samples = len(self.labels)
        
        # Inverse frequency weighting
        class_weights = total_samples / (n_classes * class_counts)
        
        return torch.FloatTensor(class_weights)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a sample"""
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image to get original size
        try:
            image = Image.open(img_path)
            original_size = image.size
        except:
            original_size = (0, 0)
        
        # Get class name
        if self.binary_classification:
            class_name = "abnormal" if label == 1 else "normal"
        else:
            class_name = self.idx_to_class[label]
        
        return {
            'image_path': img_path,
            'filename': os.path.basename(img_path),
            'label': label,
            'class_name': class_name,
            'original_size': original_size,
            'target_size': self.image_size
        }
    
    def save_split_info(self, output_dir: str):
        """Save split information for reproducibility"""
        os.makedirs(output_dir, exist_ok=True)
        
        split_info = {
            'split': self.split,
            'total_samples': len(self.samples),
            'binary_classification': self.binary_classification,
            'class_distribution': {},
            'samples': []
        }
        
        # Calculate class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if self.binary_classification:
                class_name = "abnormal" if label == 1 else "normal"
            else:
                class_name = self.idx_to_class[label]
            split_info['class_distribution'][class_name] = int(count)
        
        # Add sample details
        for idx in range(len(self.samples)):
            sample_info = self.get_sample_info(idx)
            split_info['samples'].append(sample_info)
        
        # Save to JSON
        output_file = os.path.join(output_dir, f'{self.split}_split_info.json')
        with open(output_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Split information saved to: {output_file}")


def create_herlev_splits(
    data_dir: str,
    output_dir: str,
    split_ratios: Dict[str, float] = None,
    copy_files: bool = True
):
    """
    Create train/val/test splits from Herlev dataset and optionally copy files.
    
    Args:
        data_dir: Path to Herlev dataset with class subdirectories
        output_dir: Output directory for splits
        split_ratios: Dictionary with split ratios (default: {'train': 0.7, 'val': 0.2, 'test': 0.1})
        copy_files: Whether to copy files to new directory structure
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    
    print(f"Creating Herlev dataset splits...")
    print(f"Split ratios: {split_ratios}")
    
    # Create splits
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        dataset = HerlevDataset(
            data_dir=data_dir,
            split=split,
            target_split_ratio=split_ratios,
            transform=False  # Don't apply transforms for splitting
        )
        
        # Save split information
        dataset.save_split_info(output_dir)
        
        # Copy files if requested
        if copy_files:
            split_output_dir = os.path.join(output_dir, split)
            
            # Group samples by class
            class_samples = {}
            for idx in range(len(dataset.samples)):
                info = dataset.get_sample_info(idx)
                class_name = info['class_name']
                if class_name not in class_samples:
                    class_samples[class_name] = []
                class_samples[class_name].append(info['image_path'])
            
            # Copy files maintaining class structure
            for class_name, samples in class_samples.items():
                class_output_dir = os.path.join(split_output_dir, class_name)
                os.makedirs(class_output_dir, exist_ok=True)
                
                for src_path in samples:
                    dst_path = os.path.join(class_output_dir, os.path.basename(src_path))
                    if not os.path.exists(dst_path):
                        import shutil
                        shutil.copy2(src_path, dst_path)
                
                print(f"  {class_name}: {len(samples)} files copied")
    
    print(f"\nDataset splits created successfully in: {output_dir}")


if __name__ == "__main__":
    # Example usage and testing
    
    # Test dataset loading
    data_dir = r"d:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\UncertainGuidePGU\Herlev\data\Herlev Dataset"
    
    print("=== Testing Herlev Dataset Loader ===")
    
    # Test multi-class classification
    print("\n1. Testing multi-class classification:")
    train_dataset = HerlevDataset(
        data_dir=data_dir,
        image_size=224,
        split='train',
        binary_classification=False,
        target_split_ratio={'train': 0.7, 'val': 0.2, 'test': 0.1}
    )
    
    # Test a sample
    image, label = train_dataset[0]
    info = train_dataset.get_sample_info(0)
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")
    print(f"  Class: {info['class_name']}")
    print(f"  Original size: {info['original_size']}")
    
    # Test binary classification
    print("\n2. Testing binary classification:")
    binary_dataset = HerlevDataset(
        data_dir=data_dir,
        image_size=224,
        split='train',
        binary_classification=True,
        target_split_ratio={'train': 0.7, 'val': 0.2, 'test': 0.1}
    )
    
    # Calculate class weights
    class_weights = binary_dataset.get_class_weights()
    print(f"  Binary class weights: {class_weights}")
    
    # Test different splits
    print("\n3. Testing different splits:")
    for split in ['val', 'test']:
        split_dataset = HerlevDataset(
            data_dir=data_dir,
            image_size=224,
            split=split,
            binary_classification=True,
            target_split_ratio={'train': 0.7, 'val': 0.2, 'test': 0.1}
        )
        print(f"  {split}: {len(split_dataset)} samples")
    
    print("\n=== Testing Complete ===")