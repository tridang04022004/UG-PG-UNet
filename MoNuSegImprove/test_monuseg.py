"""
Test and evaluation script for Progressive Growing U-Net on MoNuSeg dataset

This script provides comprehensive evaluation capabilities including:
- Model inference on test images
- Nuclei-specific metrics calculation
- Visualization of predictions
- Performance analysis across different stages
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UG_unet import ProgressiveUNet, PGUNet1, PGUNet2, PGUNet3, PGUNet4
from monuseg_dataset import MoNuSegDataset
from preprocessing_utils import xml_to_mask


class MoNuSegInferer:
    """Simple inference helper for single images or directories.

    Methods:
    - infer_image: run model on one image and return mask + confidence
    - infer_dir: iterate over images in a folder and save masks/visualizations
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inferer using device: {self.device}")

        # Load model using the existing evaluator loader to keep parity
        # Create a proper evaluator instance and reuse its model
        evaluator = MoNuSegEvaluator(model_path, device=self.device)
        self.model = evaluator.model
        self.model.eval()

    def infer_image(self, image_path: str, target_size: int = 256):
        """Run inference on a single image path. Returns (orig, mask, confidence)"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        image_resized = image.resize((target_size, target_size))
        image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.sigmoid(output)
            pred_mask = (probs > 0.5).float()

            # Resize back to original image size (height, width)
            pred_mask_resized = nn.functional.interpolate(pred_mask, size=original_image.shape[:2], mode='nearest')
            pred_mask_np = pred_mask_resized.squeeze().cpu().numpy()
            confidence = probs.mean().item()

        return original_image, pred_mask_np, confidence

    def infer_dir(self, input_dir: str, out_dir: str = None, target_size: int = 256):
        """Run inference for all .tif/.jpg/.png images in a directory and save masks/visuals."""
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]
        results = []
        for fname in sorted(image_files):
            p = os.path.join(input_dir, fname)
            orig, mask, conf = self.infer_image(p, target_size=target_size)
            results.append((fname, orig, mask, conf))

            if out_dir:
                # Save mask as PNG
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_mask.png")
                Image.fromarray(mask_uint8).save(mask_path)

                # Save overlay visualization
                overlay = orig.copy()
                overlay[mask > 0.5] = [255, 0, 0]
                blended = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)
                vis_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_vis.png")
                cv2.imwrite(vis_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        return results



class MoNuSegEvaluator:
    """
    Comprehensive evaluator for MoNuSeg trained models.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        print(f"Loading model from: {model_path}")
        # Load the checkpoint/state dict safely
        checkpoint = torch.load(model_path, map_location=self.device)

        # Determine whether checkpoint is a full checkpoint dict (with metadata)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            stage = checkpoint.get('stage', 4)
            state_dict = checkpoint['model_state_dict']
            print(f"Loading Stage {stage} model from checkpoint dict")
        elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            # Looks like a state_dict saved directly
            stage = 4
            state_dict = checkpoint
            print("Loading model from raw state_dict (assuming stage 4)")
        else:
            # Unknown checkpoint format
            raise RuntimeError(f"Unrecognized checkpoint format for: {model_path}")
        
        # Create the appropriate stage model (imports already available at top)
        stage_models = {
            1: PGUNet1,
            2: PGUNet2, 
            3: PGUNet3,
            4: PGUNet4
        }
        
        model_class = stage_models.get(stage, PGUNet4)  # Default to stage 4
        model = model_class(in_channels=3, num_classes=1)
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        # If checkpoint was a full dict, print extra info
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            epoch = checkpoint.get('epoch', 'unknown')
            val_dice = checkpoint.get('val_dice', 'unknown')
            print(f"Loaded model from stage {stage}, epoch {epoch}, val_dice: {val_dice}")
        else:
            print("Loaded model state dict")
        
        model.to(self.device)
        return model
    
    def predict_image(self, image_path: str, target_size: int = 256) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict nuclei mask for a single image.
        
        Args:
            image_path: Path to input image
            target_size: Size to resize image for prediction
            
        Returns:
            Tuple of (original_image, predicted_mask, confidence_score)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Resize for model input
        image_resized = image.resize((target_size, target_size))
        image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Apply sigmoid and convert to binary mask
            probs = torch.sigmoid(output)
            pred_mask = (probs > 0.5).float()
            
            # Resize back to original image size
            pred_mask_resized = nn.functional.interpolate(
                pred_mask, size=original_image.shape[:2], mode='nearest'
            )
            
            # Convert to numpy
            pred_mask_np = pred_mask_resized.squeeze().cpu().numpy()
            confidence = probs.mean().item()
        
        return original_image, pred_mask_np, confidence
    
    def evaluate_dataset(self, dataset_path: str, split: str = 'val') -> Dict[str, float]:
        """
        Evaluate model on entire dataset split.
        
        Args:
            dataset_path: Path to MoNuSeg dataset
            split: Dataset split to evaluate ('train', 'val', 'test')
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating on {split} split...")
        
        # Create dataset
        dataset = MoNuSegDataset(
            data_dir=dataset_path,
            image_size=256,
            split=split,
            transform=True,
            augment=False
        )
        
        metrics = {
            'iou': [],
            'dice': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'specificity': []
        }
        
        for i in range(len(dataset)):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(dataset)}")
            
            # Get ground truth
            image_tensor, mask_gt = dataset[i]
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            mask_gt = mask_gt.squeeze().cpu().numpy()
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                probs = torch.sigmoid(output)
                pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()
            
            # Calculate metrics
            sample_metrics = self.calculate_metrics(pred_mask, mask_gt)
            for key in metrics:
                metrics[key].append(sample_metrics[key])
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        std_metrics = {key: np.std(values) for key, values in metrics.items()}
        
        print("\n=== Evaluation Results ===")
        for key in avg_metrics:
            print(f"{key.capitalize()}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
        
        return avg_metrics, std_metrics
    
    def calculate_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive segmentation metrics"""
        # Flatten arrays
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Basic metrics
        intersection = np.sum(pred_flat * gt_flat)
        union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
        
        # True/False Positives/Negatives
        tp = intersection
        fp = np.sum(pred_flat) - tp
        fn = np.sum(gt_flat) - tp
        tn = len(pred_flat) - tp - fp - fn
        
        # Calculate metrics with epsilon to avoid division by zero
        eps = 1e-8
        
        iou = (tp + eps) / (tp + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)
        
        return {
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }
    
    def visualize_predictions(self, image_paths: List[str], output_dir: str = None):
        """
        Visualize model predictions on multiple images.
        
        Args:
            image_paths: List of paths to test images
            output_dir: Directory to save visualizations
        """
        # Import plotting here to avoid heavy imports at module import time
        import matplotlib.pyplot as plt

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {image_path}")
            
            # Get prediction
            original_image, pred_mask, confidence = self.predict_image(image_path)
            
            # Load ground truth if available
            xml_path = image_path.replace('.tif', '.xml').replace('images', 'annots')
            gt_mask = None
            if os.path.exists(xml_path):
                gt_mask = xml_to_mask(xml_path, (original_image.shape[1], original_image.shape[0]))
            
            # Create visualization
            fig, axes = plt.subplots(1, 4 if gt_mask is not None else 3, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title(f'Original Image\n{os.path.basename(image_path)}')
            axes[0].axis('off')
            
            # Predicted mask
            axes[1].imshow(pred_mask, cmap='gray')
            axes[1].set_title(f'Predicted Mask\nConfidence: {confidence:.3f}')
            axes[1].axis('off')
            
            # Overlay
            overlay = original_image.copy()
            overlay[pred_mask > 0.5] = [255, 0, 0]  # Red for predicted nuclei
            blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
            axes[2].imshow(blended)
            axes[2].set_title('Prediction Overlay\n(Red = Predicted Nuclei)')
            axes[2].axis('off')
            
            # Ground truth comparison if available
            if gt_mask is not None:
                # Calculate metrics
                metrics = self.calculate_metrics(pred_mask, gt_mask)
                
                # Create comparison overlay
                comparison = original_image.copy()
                comparison[gt_mask == 1] = [0, 255, 0]  # Green for ground truth
                comparison[pred_mask > 0.5] = [255, 0, 0]  # Red for prediction
                comparison[(gt_mask == 1) & (pred_mask > 0.5)] = [255, 255, 0]  # Yellow for overlap
                
                axes[3].imshow(comparison)
                axes[3].set_title(f'GT vs Prediction\nDice: {metrics["dice"]:.3f}, IoU: {metrics["iou"]:.3f}\n'
                                f'Green=GT, Red=Pred, Yellow=Overlap')
                axes[3].axis('off')
            
            plt.tight_layout()
            
            if output_dir:
                save_path = os.path.join(output_dir, f'prediction_{i+1:03d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization to {save_path}")
            else:
                plt.show()
    
    def test_random_images(self, dataset_path: str, num_images: int = 5, split: str = 'train'):
        """Test model on random images from dataset"""
        images_dir = os.path.join(dataset_path, split, 'images')
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
        
        # Select random images
        import random
        random.seed(42)  # For reproducibility
        selected_files = random.sample(image_files, min(num_images, len(image_files)))
        image_paths = [os.path.join(images_dir, f) for f in selected_files]
        
        print(f"Testing on {len(image_paths)} random images from {split} split")
        
        # Create output directory
        output_dir = os.path.join(dataset_path, 'test_results', 
                                 f'random_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Visualize predictions
        self.visualize_predictions(image_paths, output_dir)
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate Progressive Growing U-Net on MoNuSeg')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to MoNuSeg dataset directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--num_test', type=int, default=5,
                        help='Number of random images to test and visualize')
    parser.add_argument('--eval_full', action='store_true',
                        help='Evaluate on full dataset (time-consuming)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--infer', type=str, nargs='+', default=None,
                        help='One or more image paths to run inference on')
    parser.add_argument('--infer_dir', type=str, default=None,
                        help='Directory containing images to run inference on')
    parser.add_argument('--infer_out', type=str, default=None,
                        help='Output directory to save inference masks/visualizations')
    
    args = parser.parse_args()
    
    print("=== MoNuSeg Model Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    print("=" * 50)
    # Resolve dataset path: if user passed parent directory (e.g., '..'), try to locate MoNuSeg subfolder
    dataset_path = args.data
    expected_images = os.path.join(dataset_path, args.split, 'images')
    if not os.path.exists(expected_images):
        # Try common alternative: user passed parent folder containing MoNuSeg
        candidate = os.path.join(dataset_path, 'MoNuSeg')
        candidate_images = os.path.join(candidate, args.split, 'images')
        if os.path.exists(candidate_images):
            print(f"Note: adjusted dataset path to: {candidate}")
            dataset_path = candidate
        else:
            raise FileNotFoundError(
                f"Could not find '{args.split}/images' under dataset path '{dataset_path}'.\n"
                f"Please pass the path to the MoNuSeg dataset directory (the folder that contains 'train', 'val', 'test').\n"
                f"Example: --data ./MoNuSeg"
            )
    # Ensure subsequent code uses the resolved dataset path
    args.data = dataset_path
    # Initialize evaluator
    evaluator = MoNuSegEvaluator(args.model)
    
    # Test on random images
    print("\n1. Testing on random images...")
    output_dir = evaluator.test_random_images(args.data, args.num_test, args.split)
    print(f"Random test results saved to: {output_dir}")

    # Inference options: single images or directory
    if args.infer or args.infer_dir:
        infer_out = args.infer_out if args.infer_out else os.path.join(os.path.dirname(args.model), 'inference_results')
        inferer = MoNuSegInferer(args.model)

        if args.infer:
            for img_path in args.infer:
                print(f"Running inference on: {img_path}")
                orig, mask, conf = inferer.infer_image(img_path)
                print(f"Confidence: {conf:.4f}")
                # Save visualization if requested
                if infer_out:
                    os.makedirs(infer_out, exist_ok=True)
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path = os.path.join(infer_out, f"{base}_mask.png")
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                    overlay = orig.copy()
                    overlay[mask > 0.5] = [255, 0, 0]
                    blended = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)
                    vis_path = os.path.join(infer_out, f"{base}_vis.png")
                    cv2.imwrite(vis_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                    print(f"Saved mask and visualization to {infer_out}")

        if args.infer_dir:
            print(f"Running inference on directory: {args.infer_dir}")
            inferer.infer_dir(args.infer_dir, out_dir=infer_out)
            print(f"Saved inference results to {infer_out}")
    
    # Full dataset evaluation if requested
    if args.eval_full:
        print("\n2. Evaluating on full dataset...")
        avg_metrics, std_metrics = evaluator.evaluate_dataset(args.data, args.split)
        
        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            results_file = os.path.join(args.output, f'evaluation_results_{args.split}.json')
            
            results = {
                'model_path': args.model,
                'dataset_path': args.data,
                'split': args.split,
                'timestamp': datetime.now().isoformat(),
                'avg_metrics': avg_metrics,
                'std_metrics': std_metrics
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Full evaluation results saved to: {results_file}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()