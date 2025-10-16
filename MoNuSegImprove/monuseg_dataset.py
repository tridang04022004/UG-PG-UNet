"""
MoNuSeg Dataset Loader for Progressive Growing U-Net

This module provides a PyTorch Dataset class for loading the MoNuSeg 
(Multi-organ Nuclei Segmentation) dataset with XML polygon annotations.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List, Dict, Any
import cv2
import random


class MoNuSegDataset(Dataset):
    """
    MoNuSeg dataset loader for nuclei segmentation.
    
    The dataset contains:
    - RGB TIFF images (1000x1000)
    - XML annotations with polygon coordinates for nuclei boundaries
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        split: str = 'train',
        transform: bool = True,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Path to MoNuSeg directory
            image_size: Target size for progressive training (32, 64, 128, 256)
            split: Dataset split ('train', 'val', 'test')
            transform: Whether to apply transforms
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.split = split
        self.transform = transform
        self.augment = augment
        
        # Setup paths
        self.images_dir = os.path.join(data_dir, split, 'images')
        self.annotations_dir = os.path.join(data_dir, split, 'annots')
        
        # Build explicit list of (image_path, annotation_path) pairs by matching basenames
        image_files_all = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith('.tif')])
        samples = []
        missing_annotations = []
        for img_name in image_files_all:
            annot_name = img_name.rsplit('.', 1)[0] + '.xml'
            img_path = os.path.join(self.images_dir, img_name)
            annot_path = os.path.join(self.annotations_dir, annot_name)
            if os.path.exists(annot_path):
                samples.append((img_path, annot_path))
            else:
                missing_annotations.append(img_name)

        if len(missing_annotations) > 0:
            print(f"Warning: {len(missing_annotations)} images have no matching annotation and will be skipped\n"
                  f"Examples: {missing_annotations[:5]}")

        if len(samples) == 0:
            raise RuntimeError(f"No image-annotation pairs found in {self.images_dir} / {self.annotations_dir}")

        # Store samples and a convenience list of image filenames (for compatibility)
        self.samples = samples
        self.image_files = [os.path.basename(s[0]) for s in samples]
        self.annotation_files = [os.path.basename(s[1]) for s in samples]
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"MoNuSeg {split} dataset: {len(self.image_files)} samples")
    
    def _setup_transforms(self):
        """Setup image transforms based on current settings"""
        # Base transforms for all images
        self.base_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        # We'll apply deterministic joint augmentations manually in _apply_joint_transforms
        self.aug_transform = None
    
    def _parse_xml_annotations(self, xml_path: str, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Parse XML file and create binary mask from polygon annotations.
        
        Args:
            xml_path: Path to XML annotation file
            image_size: (width, height) of the image
            
        Returns:
            Binary mask as numpy array
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Create blank mask
        mask = np.zeros(image_size[::-1], dtype=np.uint8)  # PIL uses (W,H), numpy uses (H,W)
        
        # Find all regions (nuclei)
        regions = root.findall('.//Region')
        
        for region in regions:
            vertices = region.findall('.//Vertex')
            if len(vertices) < 3:  # Need at least 3 points for a polygon
                continue
                
            # Extract polygon coordinates
            polygon_points = []
            for vertex in vertices:
                x = float(vertex.attrib['X'])
                y = float(vertex.attrib['Y'])
                polygon_points.append((x, y))
            
            # Draw filled polygon on mask using PIL
            mask_pil = Image.fromarray(mask)
            draw = ImageDraw.Draw(mask_pil)
            draw.polygon(polygon_points, fill=1)
            mask = np.array(mask_pil)
        
        return mask
    
    def _apply_joint_transforms(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms that need to be applied to both image and mask consistently.
        """
        # Resize both image and mask first
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_resized = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Apply deterministic joint augmentations when requested
        if self.augment and self.split == 'train':
            # Use a per-sample random generator to ensure image and mask receive identical geometric transforms
            seed = torch.randint(0, 2**32, (1,)).item()
            rng = random.Random(seed)

            # Horizontal flip
            if rng.random() < 0.5:
                image_resized = TF.hflip(image_resized)
                mask_resized = TF.hflip(mask_resized)

            # Vertical flip
            if rng.random() < 0.5:
                image_resized = TF.vflip(image_resized)
                mask_resized = TF.vflip(mask_resized)

            # Rotation (apply same angle to both; mask uses NEAREST)
            angle = rng.uniform(-90, 90)
            if abs(angle) > 1e-3:
                # Use PIL.Image.rotate directly for broader compatibility with torchvision versions
                image_resized = image_resized.rotate(angle, resample=Image.BILINEAR)
                mask_resized = mask_resized.rotate(angle, resample=Image.NEAREST)

            # Random crop could be added here if desired

            # Color jitter (image only)
            if rng.random() < 0.8:
                # brightness, contrast, saturation, hue
                b = 1.0 + rng.uniform(-0.2, 0.2)
                c = 1.0 + rng.uniform(-0.2, 0.2)
                s = 1.0 + rng.uniform(-0.2, 0.2)
                h = rng.uniform(-0.05, 0.05)
                image_resized = TF.adjust_brightness(image_resized, b)
                image_resized = TF.adjust_contrast(image_resized, c)
                image_resized = TF.adjust_saturation(image_resized, s)
                image_resized = TF.adjust_hue(image_resized, h)

        # Convert to tensors
        image_tensor = transforms.ToTensor()(image_resized)
        mask_np = np.array(mask_resized)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        return image_tensor, mask_tensor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Load image and corresponding annotation from paired samples
        image_path, annotation_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        # Load and parse annotations
        mask_array = self._parse_xml_annotations(annotation_path, image.size)
        mask = Image.fromarray(mask_array)

        # Apply transforms
        if self.transform:
            image_tensor, mask_tensor = self._apply_joint_transforms(image, mask)
        else:
            # Just convert to tensors without transforms
            image_tensor = transforms.ToTensor()(image)
            mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)

        return image_tensor, mask_tensor
    
    def update_image_size(self, new_size: int):
        """Update image size for progressive training"""
        self.image_size = new_size
        self._setup_transforms()
        print(f"Updated dataset image size to {new_size}x{new_size}")
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample"""
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
        
        # Load image to get original size
        image = Image.open(image_path)
        
        # Parse XML to count nuclei
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        regions = root.findall('.//Region')
        
        return {
            'image_file': self.image_files[idx],
            'annotation_file': self.annotation_files[idx],
            'original_size': image.size,
            'num_nuclei': len(regions),
            'microns_per_pixel': float(root.attrib.get('MicronsPerPixel', 0.252))
        }


def create_train_val_split(data_dir: str, val_ratio: float = 0.2, seed: int = 42, move: bool = False):
    """
    Create train/validation split from the training data.
    Since MoNuSeg only provides a train folder, we need to split it.
    
    Args:
        data_dir: Path to MoNuSeg directory
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducible splits
    """
    import shutil
    import random
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Create validation directory structure
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'annots'), exist_ok=True)
    
    # Get all files
    image_files = sorted([f for f in os.listdir(os.path.join(train_dir, 'images')) if f.endswith('.tif')])
    
    # Split files
    random.seed(seed)
    n_val = int(len(image_files) * val_ratio)
    val_files = random.sample(image_files, n_val)
    
    action = 'move' if move else 'copy'
    print(f"{action.title()}ing {n_val} files to validation set (move={move})...")

    # Copy or move validation files
    for img_file in val_files:
        # Corresponding annotation file
        annot_file = img_file.replace('.tif', '.xml')

        # Source and destination paths
        src_img = os.path.join(train_dir, 'images', img_file)
        dst_img = os.path.join(val_dir, 'images', img_file)

        src_annot = os.path.join(train_dir, 'annots', annot_file)
        dst_annot = os.path.join(val_dir, 'annots', annot_file)

        if move:
            shutil.move(src_img, dst_img)
            if os.path.exists(src_annot):
                shutil.move(src_annot, dst_annot)
        else:
            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_annot):
                shutil.copy2(src_annot, dst_annot)
    
    print(f"Train/Val split complete:")
    print(f"  Training: {len(os.listdir(os.path.join(train_dir, 'images')))} samples")
    print(f"  Validation: {len(os.listdir(os.path.join(val_dir, 'images')))} samples")


if __name__ == "__main__":
    # Test the dataset loader
    data_dir = r"d:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSeg"
    
    # Create train/val split if validation doesn't exist
    if not os.path.exists(os.path.join(data_dir, 'val')):
        print("Creating train/validation split...")
        create_train_val_split(data_dir, val_ratio=0.2)
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    dataset = MoNuSegDataset(data_dir, image_size=256, split='train')
    
    # Test a sample
    image, mask = dataset[0]
    print(f"Sample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask range: {mask.min():.3f} - {mask.max():.3f}")
    print(f"  Nuclei pixels: {mask.sum().item():.0f}")
    
    # Get sample info
    info = dataset.get_sample_info(0)
    print(f"  Sample info: {info}")