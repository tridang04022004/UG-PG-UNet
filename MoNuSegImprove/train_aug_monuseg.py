"""
Training script for Uncertainty-Guided Progressive Growing U-Net on the augmented MoNuSeg dataset

This script implements uncertainty-guided progressive training that:
1. Uses augmented training data from the `train/aug` folder (via `AugMoNuSegDataset`)
2. Applies uncertainty-guided loss weighting from stage 2 onwards 
3. Progressively grows the network architecture through stages (32->64->128->256px)
4. Transfers weights between progressive stages

The uncertainty weighting works by:
- Computing uncertainty maps from previous stage models
- Using these maps to weight the loss function: weight = 1.0 + alpha * uncertainty
- This focuses training on uncertain regions from previous stages

Place this file in the `MoNuSegImprove` folder alongside dataset modules.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import argparse

# Ensure project root is on path so we can import UG models and trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UG_unet import ProgressiveUNet, PGUNet1, PGUNet2, PGUNet3, PGUNet4
from uncertainty_guided_trainer import UncertaintyGuidedProgressiveTrainer
from monuseg_dataset import MoNuSegDataset, create_train_val_split
from aug_monuseg_dataset import AugMoNuSegDataset


class AugMoNuSegTrainer(UncertaintyGuidedProgressiveTrainer):
    """
    Trainer that uses the augmented training dataset (train/aug) for training
    with uncertainty-guided loss weighting.
    """

    def __init__(self, config):
        self.config = config
        super().__init__(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            device=config['device'],
            uncertainty_alpha=config.get('uncertainty_alpha', 1.0)
        )

        # Override stage configs with config values
        epochs_per_stage = config.get('num_epochs_per_stage', 50)
        for stage in self.stage_configs:
            self.stage_configs[stage]['epochs_per_stage'] = epochs_per_stage

    def setup_datasets(self):
        """Setup datasets: use augmented dataset for training and standard val for validation."""
        print("Setting up augmented MoNuSeg datasets...")

        # Create train/val split if needed (operates on train/images & train/annots)
        val_dir = os.path.join(self.config['data_dir'], 'val')
        if not os.path.exists(val_dir):
            print("Creating train/validation split (will not touch train/aug)...")
            create_train_val_split(
                self.config['data_dir'],
                val_ratio=self.config.get('val_ratio', 0.2)
            )

        self.train_datasets = {}
        self.val_datasets = {}

        for stage in range(1, 5):
            image_size = self.stage_configs[stage]['resolution']

            # Training uses augmented dataset located under train/aug
            self.train_datasets[stage] = AugMoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=image_size,
                transform=True,
                augment=True
            )

            # Validation uses the standard (non-augmented) val split
            self.val_datasets[stage] = MoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=image_size,
                split='val',
                transform=True,
                augment=False
            )

        print(f"Dataset setup complete:\n  Training samples (stage1): {len(self.train_datasets[1])}\n  Validation samples (stage1): {len(self.val_datasets[1])}")

        # Compute pos_weight automatically from training masks and update base_criterion
        try:
            print("Computing positive class weight from training masks (using augmented dataset without augment)...")
            stats_ds = AugMoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=self.stage_configs[4]['resolution'],
                transform=True,
                augment=False
            )

            total_pos = 0.0
            total_pix = 0
            for i in range(len(stats_ds)):
                _, mask = stats_ds[i]
                total_pos += mask.sum().item()
                total_pix += mask.numel()

            pos_ratio = (total_pos / total_pix) if total_pix > 0 else 0.0
            computed_pos_weight = float((1.0 - pos_ratio) / (pos_ratio + 1e-8))

            # Update the base criterion with computed pos_weight
            self.base_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([computed_pos_weight]).to(self.device), 
                reduction='none'
            )
            print(f"Auto pos_weight={computed_pos_weight:.3f} (positive ratio={pos_ratio:.4f}) set for BCEWithLogitsLoss")
        except Exception as e:
            print(f"Warning: failed to compute pos_weight automatically: {e}. Using default criterion.")

    # All other methods are inherited from UncertaintyGuidedProgressiveTrainer


def create_config():
    """Create configuration that points to the MoNuSegImprove dataset root."""
    config = {
        # Data settings - point to MoNuSegImprove where train/aug lives
        'data_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\UncertainGuidePGU\MoNuSegImprove',
        'val_ratio': 0.2,

        # Model settings
        'in_channels': 3,
        'num_classes': 1,

        # Training settings
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs_per_stage': 50,
        'num_workers': 4,
        'log_interval': 10,

        # Uncertainty-guided training settings
        'uncertainty_alpha': 1.0,  # Weight factor for uncertainty weighting

        # Progressive training stages
        'stages': [1, 2, 3, 4],

        # Output settings
        'output_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\UncertainGuidePGU\MoNuSegImprove\outputs',
        'save_interval': 10,

        # Optimization settings
        'weight_decay': 1e-4,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,

        # Device settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Uncertainty-Guided Progressive Growing U-Net on augmented MoNuSeg')
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3, 4], help='Training stages to run')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per stage')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--uncertainty_alpha', type=float, default=1.0, help='Uncertainty weighting factor')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')

    args = parser.parse_args()

    config = create_config()
    config['stages'] = args.stages
    config['num_epochs_per_stage'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['uncertainty_alpha'] = args.uncertainty_alpha

    print("=== Uncertainty-Guided Progressive Growing U-Net Training (Augmented MoNuSeg) ===")
    print(f"Device: {config['device']}")
    print(f"Training stages: {config['stages']}")
    print(f"Epochs per stage: {config['num_epochs_per_stage']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Uncertainty alpha: {config['uncertainty_alpha']}")
    print("=" * 70)

    trainer = AugMoNuSegTrainer(config)
    trainer.setup_datasets()

    try:
        # Train progressively through all stages with uncertainty-guided loss
        for stage in config['stages']:
            print(f"\n=== Starting Stage {stage} Training ===")
            
            # Create data loaders for current stage
            train_loader = DataLoader(
                trainer.train_datasets[stage],
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config.get('num_workers', 4)
            )

            val_loader = DataLoader(
                trainer.val_datasets[stage],
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config.get('num_workers', 4)
            )
            
            # Set current stage and model
            trainer.current_stage = stage
            trainer.current_model = trainer.models[stage]
            trainer.setup_optimizer(stage)
            
            # Transfer weights from previous stage if not stage 1
            if stage > 1:
                trainer.transfer_weights(stage - 1, stage)
            
            # Train current stage
            for epoch in range(trainer.stage_configs[stage]['epochs_per_stage']):
                print(f"\nStage {stage}, Epoch {epoch + 1}/{trainer.stage_configs[stage]['epochs_per_stage']}")
                
                # Train one epoch
                train_loss, train_base_loss, train_dice, train_accuracy, train_unc_mean, train_unc_std = trainer.train_epoch(train_loader, stage)
                
                # Validate
                val_loss, val_base_loss, val_dice, val_accuracy, val_unc_mean, val_unc_std = trainer.validate_epoch(val_loader, stage)
                
                # Print progress
                print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train Acc: {train_accuracy:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val Acc: {val_accuracy:.4f}")
                
                if stage > 1:
                    print(f"Train Uncertainty - Mean: {train_unc_mean:.4f}, Std: {train_unc_std:.4f}")
                    print(f"Val Uncertainty - Mean: {val_unc_mean:.4f}, Std: {val_unc_std:.4f}")
                
                # Save checkpoint every few epochs
                if (epoch + 1) % config.get('save_interval', 10) == 0 or epoch == trainer.stage_configs[stage]['epochs_per_stage'] - 1:
                    checkpoint_path = os.path.join(config['output_dir'], f'pgunet_stage{stage}_epoch{epoch+1}.pth')
                    os.makedirs(config['output_dir'], exist_ok=True)
                    torch.save({
                        'model_state_dict': trainer.current_model.state_dict(),
                        'stage': stage,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_dice': train_dice,
                        'val_loss': val_loss,
                        'val_dice': val_dice,
                        'uncertainty_alpha': config['uncertainty_alpha']
                    }, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save final model for this stage
            final_path = os.path.join(config['output_dir'], f'pgunet_stage{stage}_best.pth')
            torch.save(trainer.current_model.state_dict(), final_path)
            print(f"Stage {stage} final model saved: {final_path}")
        
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
