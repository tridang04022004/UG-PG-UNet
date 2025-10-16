"""
Test and evaluation script for Herlev Cervical Cell Classification models

This script provides comprehensive evaluation capabilities including:
- Model inference on test images  
- Classification metrics calculation
- Confusion matrix and classification report
- Visualization of predictions
- Error analysis
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some visualizations may be simplified.")
from PIL import Image
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any
try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some metrics will be computed manually.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from herlev_dataset import HerlevDataset
from train_herlev import HerlevClassificationModel


class HerlevEvaluator:
    """
    Comprehensive evaluator for Herlev cervical cell classification models
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model, self.config = self.load_model(model_path)
        self.model.eval()
        
        # Get class names
        if self.config.get('binary_classification', False):
            self.class_names = ['Normal', 'Abnormal']
        else:
            self.class_names = HerlevDataset.CLASS_NAMES
        
        self.num_classes = len(self.class_names)
    
    def load_model(self, model_path: str) -> Tuple[nn.Module, Dict]:
        """Load trained model from checkpoint"""
        print(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config and model info
        config = checkpoint.get('config', {})
        stage = checkpoint.get('stage', 4)
        num_classes = config.get('num_classes', 7)
        
        print(f"Model stage: {stage}, Classes: {num_classes}")
        
        # Create model
        model = HerlevClassificationModel(
            stage=stage,
            num_classes=num_classes,
            pretrained_unet_path=None
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded successfully")
        if 'val_acc' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        return model, config
    
    def predict_single(self, image_path: str, image_size: int = None) -> Tuple[int, np.ndarray, float]:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            image_size: Target image size (if None, use model's expected size)
            
        Returns:
            Tuple of (predicted_class, probabilities, confidence)
        """
        if image_size is None:
            # Determine image size based on model stage
            stage = getattr(self.model, 'stage', 4)
            image_size = {1: 32, 2: 64, 3: 128, 4: 224}.get(stage, 224)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Build transforms matching HerlevDataset._setup_transforms
        import torchvision.transforms as transforms
        transform_list = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        # Use ImageNet normalization by default
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)
        image_tensor = transform(image)
        
        # Add batch dimension and predict
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        return predicted_class, probabilities, confidence
    
    def evaluate_dataset(self, dataset_path: str, split: str = 'test') -> Dict[str, Any]:
        """
        Evaluate model on dataset split
        
        Args:
            dataset_path: Path to Herlev dataset
            split: Dataset split to evaluate ('train', 'val', 'test')
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating on {split} split...")
        
        # Get image size for current model stage
        stage = getattr(self.model, 'stage', 4)
        image_size = {1: 32, 2: 64, 3: 128, 4: 224}.get(stage, 224)
        
        # Create dataset
        dataset = HerlevDataset(
            data_dir=dataset_path,
            image_size=image_size,
            split=split,
            transform=True,
            augment=False,
            binary_classification=self.config.get('binary_classification', False)
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"Processing {len(dataset)} samples...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(dataloader)}")
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Get predictions and probabilities
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        results = self.calculate_metrics(all_labels, all_preds, all_probs)
        results['dataset_info'] = {
            'split': split,
            'total_samples': len(all_labels),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        
        if SKLEARN_AVAILABLE:
            # Classification report
            report = classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
        else:
            # Manual implementations
            report = self._manual_classification_report(y_true, y_pred)
            cm = self._manual_confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': {}
        }
        
        # Per-class accuracy
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if class_mask.sum() > 0:
                class_acc = np.mean(y_pred[class_mask] == i)
                metrics['per_class_accuracy'][class_name] = class_acc
        
        # ROC AUC for multi-class or binary
        if SKLEARN_AVAILABLE:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                    metrics['roc_auc'] = auc
                else:
                    # Multi-class classification
                    y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                    auc_scores = []
                    for i in range(self.num_classes):
                        if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
                            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                            auc_scores.append(auc)
                    
                    if auc_scores:
                        metrics['roc_auc_per_class'] = auc_scores
                        metrics['roc_auc_macro'] = np.mean(auc_scores)
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
        
        return metrics
    
    def _manual_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Manual confusion matrix calculation"""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            if 0 <= true_label < self.num_classes and 0 <= pred_label < self.num_classes:
                cm[true_label, pred_label] += 1
        return cm
    
    def _manual_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Manual classification report calculation"""
        report = {}
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            true_pos = np.sum((y_true == i) & (y_pred == i))
            false_pos = np.sum((y_true != i) & (y_pred == i))
            false_neg = np.sum((y_true == i) & (y_pred != i))
            support = np.sum(y_true == i)
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }
        
        # Macro averages
        precisions = [report[cls]['precision'] for cls in self.class_names]
        recalls = [report[cls]['recall'] for cls in self.class_names]
        f1s = [report[cls]['f1-score'] for cls in self.class_names]
        supports = [report[cls]['support'] for cls in self.class_names]
        
        report['macro avg'] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1-score': np.mean(f1s),
            'support': np.sum(supports)
        }
        
        # Weighted averages
        total_support = np.sum(supports)
        if total_support > 0:
            report['weighted avg'] = {
                'precision': np.sum([p * s for p, s in zip(precisions, supports)]) / total_support,
                'recall': np.sum([r * s for r, s in zip(recalls, supports)]) / total_support,
                'f1-score': np.sum([f * s for f, s in zip(f1s, supports)]) / total_support,
                'support': total_support
            }
        
        return report
    
    def print_classification_report(self, results: Dict):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        
        report = results['classification_report']
        
        if SKLEARN_AVAILABLE:
            # sklearn format is already nice
            print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 75)
            
            for class_name in self.class_names:
                if class_name in report:
                    metrics = report[class_name]
                    print(f"{class_name:<25} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
            
            print("-" * 75)
            
            # Print averages
            if 'macro avg' in report:
                metrics = report['macro avg']
                print(f"{'macro avg':<25} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                      f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
            
            if 'weighted avg' in report:
                metrics = report['weighted avg']
                print(f"{'weighted avg':<25} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                      f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
        else:
            # Use manual classification report
            print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 75)
            
            # Print per-class metrics
            for class_name in self.class_names:
                if class_name in report:
                    metrics = report[class_name]
                    print(f"{class_name:<25} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
            
            print("-" * 75)
            
            # Print averages
            if 'macro avg' in report:
                metrics = report['macro avg']
                print(f"{'macro avg':<25} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                      f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
            
            if 'weighted avg' in report:
                metrics = report['weighted avg']
                print(f"{'weighted avg':<25} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                      f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
        
        print("-" * 75)
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        
        if 'roc_auc' in results:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
        elif 'roc_auc_macro' in results:
            print(f"ROC AUC (Macro): {results['roc_auc_macro']:.4f}")
        print("="*60)
    
    def visualize_confusion_matrix(self, cm: np.ndarray, output_path: str = None):
        """Visualize confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        if SEABORN_AVAILABLE:
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cbar_kws={'label': 'Normalized Count'}
            )
        else:
            # Fallback to matplotlib imshow
            im = plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, label='Normalized Count')
            
            # Add text annotations
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    plt.text(j, i, f'{cm_norm[i, j]:.2f}', 
                            ha="center", va="center", color="black")
            
            plt.xticks(range(len(self.class_names)), self.class_names)
            plt.yticks(range(len(self.class_names)), self.class_names)
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_class_distribution(self, results: Dict, output_path: str = None):
        """Visualize class distribution and performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        class_counts = []
        for class_name in self.class_names:
            if class_name in results['classification_report']:
                support = results['classification_report'][class_name]['support']
                class_counts.append(support)
            else:
                class_counts.append(0)
        
        axes[0, 0].bar(self.class_names, class_counts)
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Per-class accuracy
        class_accs = []
        for class_name in self.class_names:
            if class_name in results['per_class_accuracy']:
                class_accs.append(results['per_class_accuracy'][class_name])
            else:
                class_accs.append(0)
        
        axes[0, 1].bar(self.class_names, class_accs)
        axes[0, 1].set_title('Per-Class Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision and Recall
        precisions = []
        recalls = []
        for class_name in self.class_names:
            if class_name in results['classification_report']:
                precisions.append(results['classification_report'][class_name]['precision'])
                recalls.append(results['classification_report'][class_name]['recall'])
            else:
                precisions.append(0)
                recalls.append(0)
        
        x_pos = np.arange(len(self.class_names))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, precisions, width, label='Precision', alpha=0.8)
        axes[1, 0].bar(x_pos + width/2, recalls, width, label='Recall', alpha=0.8)
        axes[1, 0].set_title('Precision and Recall by Class')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(self.class_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # F1 scores
        f1_scores = []
        for class_name in self.class_names:
            if class_name in results['classification_report']:
                f1_scores.append(results['classification_report'][class_name]['f1-score'])
            else:
                f1_scores.append(0)
        
        axes[1, 1].bar(self.class_names, f1_scores)
        axes[1, 1].set_title('F1-Score by Class')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def predict_samples(self, dataset_path: str, num_samples: int = 10, split: str = 'test', output_dir: str = None):
        """Predict on random samples and visualize results"""
        
        # Get image size for current model stage
        stage = getattr(self.model, 'stage', 4)
        image_size = {1: 32, 2: 64, 3: 128, 4: 224}.get(stage, 224)
        
        # Create dataset
        dataset = HerlevDataset(
            data_dir=dataset_path,
            image_size=image_size,
            split=split,
            transform=False,  # We'll handle transforms manually for visualization
            augment=False,
            binary_classification=self.config.get('binary_classification', False)
        )
        
        # Select random samples
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        # Create visualization
        cols = min(5, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, sample_idx in enumerate(indices):
            row = idx // cols
            col = idx % cols
            
            # Get sample info
            sample_info = dataset.get_sample_info(sample_idx)
            image_path = sample_info['image_path']
            true_label = sample_info['label']
            true_class = sample_info['class_name']
            
            # Load and display original image
            image = Image.open(image_path).convert('RGB')
            
            # Predict
            pred_label, probs, confidence = self.predict_single(image_path, image_size)
            pred_class = self.class_names[pred_label]
            
            # Display
            if rows > 1:
                ax = axes[row][col]
            else:
                ax = axes[col] if cols > 1 else axes[0]
            
            ax.imshow(image)
            ax.axis('off')
            
            # Color code: green for correct, red for incorrect
            color = 'green' if pred_label == true_label else 'red'
            
            title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}"
            ax.set_title(title, color=color, fontsize=10)
        
        # Hide unused subplots
        for idx in range(len(indices), rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row][col].axis('off')
            else:
                if cols > 1:
                    axes[col].axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'sample_predictions_{split}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Herlev Cervical Cell Classification Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to Herlev dataset directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                       help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of sample predictions to visualize')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("=== Herlev Cervical Cell Classification Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = HerlevEvaluator(args.model, args.device)
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        model_dir = os.path.dirname(args.model)
        output_dir = os.path.join(model_dir, f'evaluation_results_{args.split}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on dataset
    print(f"\n1. Evaluating on {args.split} split...")
    results = evaluator.evaluate_dataset(args.data, args.split)
    
    # Print detailed results
    evaluator.print_classification_report(results)
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = results.copy()
    results_json['confusion_matrix'] = np.array(results['confusion_matrix']).tolist()
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Generate visualizations
    print("\n2. Generating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    evaluator.visualize_confusion_matrix(np.array(results['confusion_matrix']), cm_path)
    
    # Class distribution and metrics
    metrics_path = os.path.join(output_dir, 'class_metrics.png')
    evaluator.visualize_class_distribution(results, metrics_path)
    
    # Sample predictions
    print(f"\n3. Visualizing {args.num_samples} sample predictions...")
    evaluator.predict_samples(args.data, args.num_samples, args.split, output_dir)
    
    # Print class-wise performance
    print(f"\nClass-wise Performance:")
    for class_name in evaluator.class_names:
        if class_name in results['classification_report']:
            metrics = results['classification_report'][class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-Score: {metrics['f1-score']:.3f}")
            print(f"    Support: {metrics['support']}")
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()