"""
Augmented MoNuSeg Dataset Loader

This module provides a PyTorch Dataset class for loading the augmented
MoNuSeg dataset located under train/aug (images and annotations). It keeps
the same API and behavior as MoNuSegDataset in monuseg_dataset.py so it can
be dropped into training code that expects (image_tensor, mask_tensor).
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
import random


class AugMoNuSegDataset(Dataset):
    """
    Dataset loader for the augmented MoNuSeg data stored in:
      <data_dir>/train/aug/images
      <data_dir>/train/aug/annots

    It mirrors the API and behavior of MoNuSegDataset in monuseg_dataset.py.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        transform: bool = True,
        augment: bool = True,
    ):
        """
        Args:
            data_dir: Path to the dataset root (the code will look under data_dir/train/aug)
            image_size: Target size for progressive training (32,64,128,256)
            transform: Whether to apply transforms
            augment: Whether to apply data augmentation (geometric/color jitter)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.augment = augment

        # Explicitly point to the augmented train folder
        self.images_dir = os.path.join(data_dir, 'train', 'aug', 'images')
        self.annotations_dir = os.path.join(data_dir, 'train', 'aug', 'annots')

        # Build list of (image_path, annotation_path) pairs by matching basenames
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

        self.samples = samples
        self.image_files = [os.path.basename(s[0]) for s in samples]
        self.annotation_files = [os.path.basename(s[1]) for s in samples]

        # Setup transforms
        self._setup_transforms()

        print(f"Augmented MoNuSeg (train/aug) dataset: {len(self.image_files)} samples")

    def _setup_transforms(self):
        self.base_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def _parse_xml_annotations(self, xml_path: str, image_size: Tuple[int, int]) -> np.ndarray:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        regions = root.findall('.//Region')

        for region in regions:
            vertices = region.findall('.//Vertex')
            if len(vertices) < 3:
                continue
            polygon_points = []
            for vertex in vertices:
                x = float(vertex.attrib['X'])
                y = float(vertex.attrib['Y'])
                polygon_points.append((x, y))

            mask_pil = Image.fromarray(mask)
            draw = ImageDraw.Draw(mask_pil)
            draw.polygon(polygon_points, fill=1)
            mask = np.array(mask_pil)

        return mask

    def _apply_joint_transforms(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_resized = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            rng = random.Random(seed)

            if rng.random() < 0.5:
                image_resized = TF.hflip(image_resized)
                mask_resized = TF.hflip(mask_resized)

            if rng.random() < 0.5:
                image_resized = TF.vflip(image_resized)
                mask_resized = TF.vflip(mask_resized)

            angle = rng.uniform(-90, 90)
            if abs(angle) > 1e-3:
                image_resized = image_resized.rotate(angle, resample=Image.BILINEAR)
                mask_resized = mask_resized.rotate(angle, resample=Image.NEAREST)

            if rng.random() < 0.8:
                b = 1.0 + rng.uniform(-0.2, 0.2)
                c = 1.0 + rng.uniform(-0.2, 0.2)
                s = 1.0 + rng.uniform(-0.2, 0.2)
                h = rng.uniform(-0.05, 0.05)
                image_resized = TF.adjust_brightness(image_resized, b)
                image_resized = TF.adjust_contrast(image_resized, c)
                image_resized = TF.adjust_saturation(image_resized, s)
                image_resized = TF.adjust_hue(image_resized, h)

        image_tensor = transforms.ToTensor()(image_resized)
        mask_np = np.array(mask_resized)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, annotation_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        mask_array = self._parse_xml_annotations(annotation_path, image.size)
        mask = Image.fromarray(mask_array)

        if self.transform:
            image_tensor, mask_tensor = self._apply_joint_transforms(image, mask)
        else:
            image_tensor = transforms.ToTensor()(image)
            mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)

        return image_tensor, mask_tensor

    def update_image_size(self, new_size: int):
        self.image_size = new_size
        self._setup_transforms()
        print(f"Updated dataset image size to {new_size}x{new_size}")

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])

        image = Image.open(image_path)
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


if __name__ == "__main__":
    # Smoke test the augmented dataset loader
    data_dir = r"d:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSegImprove"
    print("Testing AugMoNuSegDataset (train/aug)...")
    dataset = AugMoNuSegDataset(data_dir, image_size=256)

    image, mask = dataset[0]
    print(f"Sample 0: Image shape: {image.shape}, Mask shape: {mask.shape}")
    info = dataset.get_sample_info(0)
    print(f"Sample info: {info}")
