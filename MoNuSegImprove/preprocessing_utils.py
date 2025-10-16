"""
Data preprocessing utilities for MoNuSeg dataset

This module provides utilities for:
- Converting XML annotations to binary masks
- Data augmentation for nuclei segmentation
- Dataset visualization and analysis
- Data quality checks
"""

import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import torch
from typing import List, Tuple, Dict, Any
import random
from pathlib import Path


def xml_to_mask(xml_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert XML annotation file to binary segmentation mask.
    
    Args:
        xml_path: Path to XML annotation file
        image_size: (width, height) of the image
        
    Returns:
        Binary mask as numpy array (H, W)
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


def analyze_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Analyze the MoNuSeg dataset to understand its characteristics.
    
    Args:
        data_dir: Path to MoNuSeg directory
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'num_images': 0,
        'image_sizes': [],
        'nuclei_counts': [],
        'nuclei_areas': [],
        'mask_coverage': [],  # Percentage of pixels that are nuclei
    }
    
    train_dir = os.path.join(data_dir, 'train')
    images_dir = os.path.join(train_dir, 'images')
    annots_dir = os.path.join(train_dir, 'annots')
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    stats['num_images'] = len(image_files)
    
    print(f"Analyzing {len(image_files)} images...")
    
    for i, img_file in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(image_files)}")
            
        # Load image
        img_path = os.path.join(images_dir, img_file)
        image = Image.open(img_path)
        stats['image_sizes'].append(image.size)
        
        # Parse XML and create mask
        xml_file = img_file.replace('.tif', '.xml')
        xml_path = os.path.join(annots_dir, xml_file)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        regions = root.findall('.//Region')
        
        # Count nuclei
        stats['nuclei_counts'].append(len(regions))
        
        # Create mask and calculate coverage
        mask = xml_to_mask(xml_path, image.size)
        total_pixels = mask.size
        nuclei_pixels = np.sum(mask)
        coverage = (nuclei_pixels / total_pixels) * 100
        stats['mask_coverage'].append(coverage)
        
        # Calculate individual nuclei areas
        for region in regions:
            area = float(region.attrib.get('Area', 0))
            if area > 0:
                stats['nuclei_areas'].append(area)
    
    # Calculate summary statistics
    stats['summary'] = {
        'avg_nuclei_per_image': np.mean(stats['nuclei_counts']),
        'std_nuclei_per_image': np.std(stats['nuclei_counts']),
        'min_nuclei_per_image': np.min(stats['nuclei_counts']),
        'max_nuclei_per_image': np.max(stats['nuclei_counts']),
        'avg_mask_coverage': np.mean(stats['mask_coverage']),
        'std_mask_coverage': np.std(stats['mask_coverage']),
        'avg_nuclei_area': np.mean(stats['nuclei_areas']),
        'std_nuclei_area': np.std(stats['nuclei_areas']),
        'total_nuclei': np.sum(stats['nuclei_counts'])
    }
    
    return stats


def visualize_samples(data_dir: str, num_samples: int = 4, save_path: str = None):
    """
    Visualize random samples from the dataset with their masks.
    
    Args:
        data_dir: Path to MoNuSeg directory
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    train_dir = os.path.join(data_dir, 'train')
    images_dir = os.path.join(train_dir, 'images')
    annots_dir = os.path.join(train_dir, 'annots')
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    
    # Select random samples
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Import plotting locally to avoid heavy imports during module import
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_file in enumerate(sample_files):
        # Load image
        img_path = os.path.join(images_dir, img_file)
        image = Image.open(img_path)
        
        # Load mask
        xml_file = img_file.replace('.tif', '.xml')
        xml_path = os.path.join(annots_dir, xml_file)
        mask = xml_to_mask(xml_path, image.size)
        
        # Create overlay
        image_np = np.array(image)
        overlay = image_np.copy()
        overlay[mask == 1] = [255, 0, 0]  # Red nuclei
        blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)
        
        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image: {img_file}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Nuclei Mask ({np.sum(mask)} pixels)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(blended)
        axes[i, 2].set_title('Overlay (Red = Nuclei)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def create_data_quality_report(data_dir: str, output_path: str = None):
    """
    Create a comprehensive data quality report for the MoNuSeg dataset.
    
    Args:
        data_dir: Path to MoNuSeg directory
        output_path: Path to save the report
    """
    print("Creating data quality report...")
    
    # Analyze dataset
    stats = analyze_dataset(data_dir)
    
    # Create plots
    # Import plotting locally to avoid heavy imports during module import
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Nuclei count distribution
    axes[0, 0].hist(stats['nuclei_counts'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Nuclei Count per Image')
    axes[0, 0].set_xlabel('Number of Nuclei')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(stats['summary']['avg_nuclei_per_image'], color='red', 
                       linestyle='--', label=f"Mean: {stats['summary']['avg_nuclei_per_image']:.1f}")
    axes[0, 0].legend()
    
    # Plot 2: Mask coverage distribution
    axes[0, 1].hist(stats['mask_coverage'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Mask Coverage (%)')
    axes[0, 1].set_xlabel('Mask Coverage (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(stats['summary']['avg_mask_coverage'], color='red', 
                       linestyle='--', label=f"Mean: {stats['summary']['avg_mask_coverage']:.1f}%")
    axes[0, 1].legend()
    
    # Plot 3: Nuclei area distribution (log scale)
    axes[0, 2].hist(np.log10(stats['nuclei_areas']), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Distribution of Nuclei Areas (log10)')
    axes[0, 2].set_xlabel('log10(Area)')
    axes[0, 2].set_ylabel('Frequency')
    
    # Plot 4: Nuclei count vs coverage scatter
    axes[1, 0].scatter(stats['nuclei_counts'], stats['mask_coverage'], alpha=0.6)
    axes[1, 0].set_title('Nuclei Count vs Mask Coverage')
    axes[1, 0].set_xlabel('Number of Nuclei')
    axes[1, 0].set_ylabel('Mask Coverage (%)')
    
    # Plot 5: Summary statistics text
    axes[1, 1].axis('off')
    summary_text = f"""
    Dataset Summary Statistics:
    
    Total Images: {stats['num_images']}
    Total Nuclei: {stats['summary']['total_nuclei']}
    
    Nuclei per Image:
    • Mean: {stats['summary']['avg_nuclei_per_image']:.1f} ± {stats['summary']['std_nuclei_per_image']:.1f}
    • Range: {stats['summary']['min_nuclei_per_image']} - {stats['summary']['max_nuclei_per_image']}
    
    Mask Coverage:
    • Mean: {stats['summary']['avg_mask_coverage']:.1f}% ± {stats['summary']['std_mask_coverage']:.1f}%
    
    Nuclei Area:
    • Mean: {stats['summary']['avg_nuclei_area']:.1f} ± {stats['summary']['std_nuclei_area']:.1f}
    """
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    # Plot 6: Image size consistency check
    unique_sizes = list(set(stats['image_sizes']))
    axes[1, 2].axis('off')
    size_text = f"Image Sizes:\n"
    for size in unique_sizes:
        count = stats['image_sizes'].count(size)
        size_text += f"• {size[0]}x{size[1]}: {count} images\n"
    axes[1, 2].text(0.05, 0.95, size_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('MoNuSeg Dataset Quality Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Data quality report saved to {output_path}")
    else:
        plt.show()
    
    return stats


def check_data_integrity(data_dir: str) -> bool:
    """
    Check data integrity for the MoNuSeg dataset.
    
    Args:
        data_dir: Path to MoNuSeg directory
        
    Returns:
        True if all checks pass, False otherwise
    """
    print("Checking data integrity...")
    
    issues = []
    
    # Check directory structure
    train_dir = os.path.join(data_dir, 'train')
    images_dir = os.path.join(train_dir, 'images')
    annots_dir = os.path.join(train_dir, 'annots')
    
    if not os.path.exists(images_dir):
        issues.append(f"Images directory not found: {images_dir}")
    if not os.path.exists(annots_dir):
        issues.append(f"Annotations directory not found: {annots_dir}")
    
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return False
    
    # Check file correspondence
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    annot_files = sorted([f for f in os.listdir(annots_dir) if f.endswith('.xml')])
    
    if len(image_files) != len(annot_files):
        issues.append(f"Mismatch: {len(image_files)} images vs {len(annot_files)} annotations")
    
    # Check file correspondence
    for img_file in image_files:
        expected_xml = img_file.replace('.tif', '.xml')
        if expected_xml not in annot_files:
            issues.append(f"Missing annotation for image: {img_file}")
    
    # Check file readability
    corrupted_files = []
    for i, img_file in enumerate(image_files[:10]):  # Check first 10 files
        try:
            # Test image loading
            img_path = os.path.join(images_dir, img_file)
            Image.open(img_path)
            
            # Test XML parsing
            xml_file = img_file.replace('.tif', '.xml')
            xml_path = os.path.join(annots_dir, xml_file)
            ET.parse(xml_path)
            
        except Exception as e:
            corrupted_files.append(f"{img_file}: {str(e)}")
    
    if corrupted_files:
        issues.extend(corrupted_files)
    
    # Report results
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Data integrity check passed!")
        print(f"  - {len(image_files)} image-annotation pairs found")
        print(f"  - All files are readable")
        return True


if __name__ == "__main__":
    # Example usage
    data_dir = r"d:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSeg"
    
    print("=== MoNuSeg Data Preprocessing Utilities ===")
    
    # Check data integrity
    if check_data_integrity(data_dir):
        
        # Create data quality report
        output_dir = os.path.join(data_dir, 'analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'data_quality_report.png')
        stats = create_data_quality_report(data_dir, report_path)
        
        # Visualize samples
        samples_path = os.path.join(output_dir, 'sample_visualization.png')
        visualize_samples(data_dir, num_samples=4, save_path=samples_path)
        
        print("\n=== Analysis Complete ===")
        print(f"Reports saved to: {output_dir}")
    else:
        print("Please fix data integrity issues before proceeding.")