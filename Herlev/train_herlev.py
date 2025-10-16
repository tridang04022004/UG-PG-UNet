"""
Training script for Uncertainty-Guided Progressive U-Net on Herlev Cervical Cell Dataset

This script implements uncertainty-guided progressive training for cervical cell classification
using the Herlev dataset. It supports both binary (normal vs abnormal) and multi-class
classification tasks.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UG_unet import ProgressiveUNet, PGUNet1, PGUNet2, PGUNet3, PGUNet4, UncertaintyGuidedLoss
from uncertainty_guided_trainer import UncertaintyGuidedProgressiveTrainer
from herlev_dataset import HerlevDataset


class HerlevClassificationModel(nn.Module):
    """
    Adaptation of Progressive U-Net for classification tasks.
    Uses the encoder part of U-Net as feature extractor followed by classification head.
    """
    
    def __init__(self, stage: int, num_classes: int, pretrained_unet_path: str = None):
        super().__init__()
        self.stage = stage
        self.num_classes = num_classes
        
        # Load the appropriate U-Net stage
        stage_models = {
            1: PGUNet1,
            2: PGUNet2,
            3: PGUNet3, 
            4: PGUNet4
        }
        
        self.unet = stage_models[stage](in_channels=3, num_classes=1)
        
        # Load pretrained weights if provided
        if pretrained_unet_path and os.path.exists(pretrained_unet_path):
            print(f"Loading pretrained U-Net weights from: {pretrained_unet_path}")
            state_dict = torch.load(pretrained_unet_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.unet.load_state_dict(state_dict)
        
        # Get feature dimensions by running a forward pass
        with torch.no_grad():
            resolutions = {1: 32, 2: 64, 3: 128, 4: 256}
            test_input = torch.randn(1, 3, resolutions[stage], resolutions[stage])
            features = self._extract_features(test_input)
            feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Freeze U-Net encoder if using pretrained weights
        if pretrained_unet_path:
            self._freeze_encoder()
    
    def _extract_features(self, x):
        """Extract features from U-Net encoder"""
        if self.stage == 1:
            x1 = self.unet.inc(x)
            return x1
        elif self.stage == 2:
            x1 = self.unet.inc(x)
            x2 = self.unet.down3(x1)
            return x2
        elif self.stage == 3:
            x1 = self.unet.inc(x)
            x2 = self.unet.down2(x1)
            x3 = self.unet.down3(x2)
            return x3
        elif self.stage == 4:
            x1 = self.unet.inc(x)
            x2 = self.unet.down1(x1)
            x3 = self.unet.down2(x2)
            x4 = self.unet.down3(x3)
            return x4
    
    def _freeze_encoder(self):
        """Freeze encoder weights for transfer learning"""
        for param in self.unet.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.unet.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # Extract features using U-Net encoder
        features = self._extract_features(x)
        
        # Classify
        output = self.classifier(features)
        
        return output


class HerlevTrainer:
    """
    Progressive trainer for Herlev cervical cell classification
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # Stage configurations matching U-Net resolutions
        self.stage_configs = {
            1: {'resolution': 32, 'epochs': config['epochs_per_stage'], 'lr': 3e-4},
            2: {'resolution': 64, 'epochs': config['epochs_per_stage'], 'lr': 1e-4},
            3: {'resolution': 128, 'epochs': config['epochs_per_stage'], 'lr': 1e-4},
            4: {'resolution': 224, 'epochs': config['epochs_per_stage'], 'lr': 1e-4}  # Use 224 for better feature learning
        }
        
        self.current_stage = 1
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
        # Initialize models for each stage
        for stage in range(1, 5):
            self.models[stage] = HerlevClassificationModel(
                stage=stage,
                num_classes=config['num_classes'],
                pretrained_unet_path=config.get('pretrained_unet_paths', {}).get(stage)
            ).to(self.device)
        
        # Loss function with class weighting
        self.setup_loss_function()
        
        # Initialize uncertainty-guided loss for classification
        self.uncertainty_loss = UncertaintyGuidedLoss(self.device)
        self.uncertainty_alpha = config.get('uncertainty_alpha', 1.0)
        
        # Training history with uncertainty metrics
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'uncertainty_weights_mean': [], 'uncertainty_weights_std': [],
            'base_loss': [],
            'stage_transitions': []
        }
    
    def setup_loss_function(self):
        """Setup loss function with class weighting"""
        if self.config.get('class_weights') is not None:
            class_weights = torch.FloatTensor(self.config['class_weights']).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def setup_optimizer_scheduler(self, stage):
        """Setup optimizer and scheduler for current stage"""
        model = self.models[stage]
        lr = self.stage_configs[stage]['lr']
        
        self.optimizers[stage] = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        self.schedulers[stage] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers[stage],
            mode='min',
            factor=0.5,
            patience=5
        )
    
    def transfer_weights(self, prev_stage, current_stage):
        """Transfer compatible weights from previous stage"""
        print(f"Transferring weights from stage {prev_stage} to {current_stage}")
        
        prev_model = self.models[prev_stage]
        current_model = self.models[current_stage]
        
        # Transfer classifier weights if dimensions match
        prev_classifier = prev_model.classifier
        current_classifier = current_model.classifier
        
        # Transfer what we can
        with torch.no_grad():
            for (prev_name, prev_param), (curr_name, curr_param) in zip(
                prev_classifier.named_parameters(), current_classifier.named_parameters()
            ):
                if prev_param.shape == curr_param.shape:
                    curr_param.copy_(prev_param)
                    print(f"  Transferred {curr_name}")
    
    def uncertainty_guided_forward_pass(self, data, target, stage):
        """
        Perform forward pass with uncertainty-guided loss weighting for classification
        
        Args:
            data: Input batch (B, C, H, W)
            target: Target batch (B,) - class indices 
            stage: Current training stage (1, 2, 3, or 4)
        
        Returns:
            final_loss: Weighted loss for backpropagation
            metrics: Dictionary with loss and uncertainty metrics
        """
        current_model = self.models[stage]
        
        # Forward pass through current model
        output_current = current_model(data)
        
        # Generate uncertainty map if not stage 1
        uncertainty_map = None
        if stage > 1:
            prev_model = self.models[stage - 1]
            prev_model.eval()  # Ensure previous model is in eval mode
            
            current_resolution = self.stage_configs[stage]['resolution']
            prev_resolution = self.stage_configs[stage - 1]['resolution']
            
            # Get uncertainty from previous stage's classifier outputs
            with torch.no_grad():
                # Resize input to previous resolution
                data_prev = F.interpolate(data, size=(prev_resolution, prev_resolution), 
                                        mode='bilinear', align_corners=True)
                
                # Get previous stage output (logits)
                output_prev = prev_model(data_prev)
                
                # Convert to probabilities using softmax for multi-class
                if self.config['num_classes'] > 2:
                    probs_prev = F.softmax(output_prev, dim=1)
                    # Calculate entropy-based uncertainty: H(p) / log(K)
                    entropy = -torch.sum(probs_prev * torch.log(probs_prev + 1e-8), dim=1, keepdim=True)
                    uncertainty_map = entropy / np.log(self.config['num_classes'])
                else:
                    # Binary classification: use sigmoid + binary entropy
                    probs_prev = torch.sigmoid(output_prev)
                    uncertainty_map = 1.0 - 2.0 * torch.abs(probs_prev - 0.5)
                
                # Resize uncertainty map to current resolution if needed
                # For classification, we typically use a single uncertainty value per sample
                # But we can also create a spatial uncertainty map
                uncertainty_map = torch.mean(uncertainty_map, dim=[2, 3], keepdim=True) if len(uncertainty_map.shape) > 2 else uncertainty_map
        
        # Calculate base loss (CrossEntropyLoss)
        base_loss = self.criterion(output_current, target)
        
        # Apply uncertainty weighting for classification
        if uncertainty_map is not None:
            # For classification, apply uncertainty as sample-wise weights
            uncertainty_weights = 1.0 + self.uncertainty_alpha * uncertainty_map.squeeze()
            if len(uncertainty_weights.shape) == 0:  # Single sample
                uncertainty_weights = uncertainty_weights.unsqueeze(0)
            
            # Create loss function that can handle sample weights
            base_losses = F.cross_entropy(output_current, target, reduction='none')
            weighted_losses = base_losses * uncertainty_weights.detach()
            final_loss = torch.mean(weighted_losses)
        else:
            # Stage 1: No uncertainty weighting
            final_loss = base_loss
            uncertainty_weights = torch.ones(data.size(0), device=self.device)
        
        # Prepare metrics
        metrics = {
            'final_loss': final_loss.item(),
            'base_loss': base_loss.item(),
            'output': output_current,
            'uncertainty_weight_mean': torch.mean(uncertainty_weights).item() if uncertainty_map is not None else 0.0,
            'uncertainty_weight_std': torch.std(uncertainty_weights).item() if uncertainty_map is not None else 0.0
        }
        
        return final_loss, metrics
    
    def train_epoch(self, dataloader, stage):
        """Train for one epoch with uncertainty-guided loss weighting"""
        model = self.models[stage]
        optimizer = self.optimizers[stage]
        model.train()
        
        # Set previous model to eval mode if exists
        if stage > 1:
            self.models[stage - 1].eval()
        
        total_loss = 0
        total_base_loss = 0
        total_uncertainty_mean = 0
        total_uncertainty_std = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with uncertainty-guided loss
            final_loss, metrics = self.uncertainty_guided_forward_pass(data, target, stage)
            
            # Backward pass
            final_loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            output = metrics['output']
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Accumulate metrics
            total_loss += metrics['final_loss']
            total_base_loss += metrics['base_loss']
            total_uncertainty_mean += metrics['uncertainty_weight_mean']
            total_uncertainty_std += metrics['uncertainty_weight_std']
            
            if batch_idx % self.config.get('log_interval', 10) == 0:
                uncertainty_info = ""
                if stage > 1:
                    uncertainty_info = f", Unc_mean: {metrics['uncertainty_weight_mean']:.4f}, Unc_std: {metrics['uncertainty_weight_std']:.4f}"
                
                print(f'Stage {stage}, Batch {batch_idx}/{len(dataloader)} '
                      f'Loss: {metrics["final_loss"]:.6f} '
                      f'Base_Loss: {metrics["base_loss"]:.6f} '
                      f'Acc: {100. * correct / total:.2f}%{uncertainty_info}')
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_base_loss = total_base_loss / num_batches
        avg_uncertainty_mean = total_uncertainty_mean / num_batches
        avg_uncertainty_std = total_uncertainty_std / num_batches
        accuracy = 100. * correct / total
        
        return avg_loss, avg_base_loss, accuracy, avg_uncertainty_mean, avg_uncertainty_std
    
    def validate_epoch(self, dataloader, stage):
        """Validate for one epoch with uncertainty metrics"""
        model = self.models[stage]
        model.eval()
        
        # Set previous model to eval mode if exists
        if stage > 1:
            self.models[stage - 1].eval()
        
        total_loss = 0
        total_base_loss = 0
        total_uncertainty_mean = 0
        total_uncertainty_std = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass with uncertainty-guided loss
                final_loss, metrics = self.uncertainty_guided_forward_pass(data, target, stage)
                
                # Calculate accuracy
                output = metrics['output']
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Accumulate metrics
                total_loss += metrics['final_loss']
                total_base_loss += metrics['base_loss']
                total_uncertainty_mean += metrics['uncertainty_weight_mean']
                total_uncertainty_std += metrics['uncertainty_weight_std']
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_base_loss = total_base_loss / num_batches
        avg_uncertainty_mean = total_uncertainty_mean / num_batches
        avg_uncertainty_std = total_uncertainty_std / num_batches
        accuracy = 100. * correct / total
        
        return avg_loss, avg_base_loss, accuracy, avg_uncertainty_mean, avg_uncertainty_std
    
    def train_progressive(self, train_loaders, val_loaders, save_dir):
        """Train progressively through all stages"""
        os.makedirs(save_dir, exist_ok=True)
        
        for stage in self.config['stages']:
            print(f"\n{'='*60}")
            print(f"Training Stage {stage} - Resolution: {self.stage_configs[stage]['resolution']}")
            print(f"{'='*60}")
            
            self.current_stage = stage
            self.setup_optimizer_scheduler(stage)
            
            # Transfer weights from previous stage
            if stage > 1 and (stage - 1) in self.models:
                self.transfer_weights(stage - 1, stage)
            
            # Get data loaders for current stage
            train_loader = train_loaders[stage]
            val_loader = val_loaders[stage]
            
            best_val_loss = float('inf')
            best_val_acc = 0
            epochs_without_improvement = 0
            
            # Training loop
            for epoch in range(self.stage_configs[stage]['epochs']):
                print(f"\nStage {stage}, Epoch {epoch + 1}/{self.stage_configs[stage]['epochs']}")
                
                # Train with uncertainty metrics
                train_loss, train_base_loss, train_acc, train_unc_mean, train_unc_std = self.train_epoch(train_loader, stage)
                
                # Validate with uncertainty metrics
                val_loss, val_base_loss, val_acc, val_unc_mean, val_unc_std = self.validate_epoch(val_loader, stage)
                
                # Update scheduler
                self.schedulers[stage].step(val_loss)
                
                # Save metrics
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                self.history['uncertainty_weights_mean'].append(val_unc_mean)
                self.history['uncertainty_weights_std'].append(val_unc_std)
                self.history['base_loss'].append(val_base_loss)
                
                # Print progress
                print(f"Train Loss: {train_loss:.4f}, Base Loss: {train_base_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Base Loss: {val_base_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                if stage > 1:
                    print(f"Train Uncertainty - Mean: {train_unc_mean:.4f}, Std: {train_unc_std:.4f}")
                    print(f"Val Uncertainty - Mean: {val_unc_mean:.4f}, Std: {val_unc_std:.4f}")
                
                # Save best model
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                    
                    # Save checkpoint
                    checkpoint = {
                        'model_state_dict': self.models[stage].state_dict(),
                        'optimizer_state_dict': self.optimizers[stage].state_dict(),
                        'stage': stage,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'config': self.config
                    }
                    
                    checkpoint_path = os.path.join(save_dir, f'herlev_stage{stage}_best.pth')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"New best model saved! Val Acc: {val_acc:.2f}%")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= self.config.get('early_stopping_patience', 15):
                    print(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                    break
            
            # Record stage completion
            self.history['stage_transitions'].append({
                'stage': stage,
                'completed_at': datetime.now().isoformat(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss
            })
            
            print(f"Stage {stage} completed. Best Val Acc: {best_val_acc:.2f}%")
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")


def create_config():
    """Create default configuration"""
    config = {
        # Data settings
        'data_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\UncertainGuidePGU\Herlev\data\Herlev Dataset',
        'binary_classification': False,  # Set to True for binary classification
        
        # Model settings
        'num_classes': 7,  # 7 for multi-class, 2 for binary
        'pretrained_unet_paths': {
            # Paths to pretrained U-Net models (optional)
            # 1: 'path/to/unet_stage1.pth',
            # 2: 'path/to/unet_stage2.pth',
            # 3: 'path/to/unet_stage3.pth', 
            # 4: 'path/to/unet_stage4.pth'
        },
        
        # Training settings
        'batch_size': 16,
        'epochs_per_stage': 30,
        'stages': [1, 2, 3, 4],
        'num_workers': 4,
        'log_interval': 10,
        'early_stopping_patience': 15,
        
        # Optimization settings
        'weight_decay': 1e-4,
        'class_weights': None,  # Will be computed automatically
        
        # Uncertainty-guided training settings
        'uncertainty_alpha': 1.0,  # Weight factor for uncertainty weighting
        
        # Split settings
        'split_ratios': {'train': 0.7, 'val': 0.2, 'test': 0.1},
        
        # Output settings
        'output_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\UncertainGuidePGU\Herlev\outputs',
        
        # Device settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    return config


def setup_datasets(config):
    """Setup datasets for all stages"""
    datasets = {}
    dataloaders = {}
    
    # Create datasets for each split
    for split in ['train', 'val']:
        datasets[split] = {}
        dataloaders[split] = {}
        
        # Create datasets for each stage with appropriate image sizes
        for stage in range(1, 5):
            image_size = {1: 32, 2: 64, 3: 128, 4: 224}[stage]  # Use 224 for stage 4
            
            datasets[split][stage] = HerlevDataset(
                data_dir=config['data_dir'],
                image_size=image_size,
                split=split,
                transform=True,
                augment=(split == 'train'),
                binary_classification=config['binary_classification'],
                target_split_ratio=config['split_ratios']
            )
            
            dataloaders[split][stage] = DataLoader(
                datasets[split][stage],
                batch_size=config['batch_size'],
                shuffle=(split == 'train'),
                num_workers=config['num_workers'],
                pin_memory=True
            )
    
    # Compute class weights from training data
    if config['class_weights'] is None:
        class_weights = datasets['train'][1].get_class_weights()  # Use stage 1 for weight calculation
        config['class_weights'] = class_weights.tolist()
        print(f"Computed class weights: {config['class_weights']}")
    
    return dataloaders['train'], dataloaders['val']


def main():
    parser = argparse.ArgumentParser(description='Train Progressive U-Net for Herlev Cervical Cell Classification')
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3, 4], help='Training stages to run')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs per stage')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--binary', action='store_true', help='Use binary classification (normal vs abnormal)')
    parser.add_argument('--uncertainty_alpha', type=float, default=1.0, help='Uncertainty weighting factor')
    parser.add_argument('--data_dir', type=str, help='Path to Herlev dataset directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config()
    config['stages'] = args.stages
    config['epochs_per_stage'] = args.epochs
    config['batch_size'] = args.batch_size
    config['binary_classification'] = args.binary
    config['uncertainty_alpha'] = args.uncertainty_alpha
    
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Update num_classes based on classification mode
    if config['binary_classification']:
        config['num_classes'] = 2
    
    print("=== Herlev Cervical Cell Classification Training ===")
    print(f"Device: {config['device']}")
    print(f"Classification mode: {'Binary' if config['binary_classification'] else 'Multi-class'}")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Training stages: {config['stages']}")
    print(f"Epochs per stage: {config['epochs_per_stage']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Uncertainty alpha: {config['uncertainty_alpha']}")
    print("=" * 60)
    
    # Setup datasets
    print("Setting up datasets...")
    train_loaders, val_loaders = setup_datasets(config)
    
    # Initialize trainer
    trainer = HerlevTrainer(config)
    
    try:
        # Train progressively
        trainer.train_progressive(train_loaders, val_loaders, config['output_dir'])
        print("Training completed successfully!")
        
        # Save final configuration
        config_path = os.path.join(config['output_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()