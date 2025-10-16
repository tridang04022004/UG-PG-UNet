"""
Demonstration script for Uncertainty-Guided Progressive Growing U-Net

This script shows how to use the newly implemented uncertainty-guided loss weighting
functionality for progressive training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

# Import the uncertainty-guided trainer
from UncertainGuidePGU.uncertainty_guided_trainer import UncertaintyGuidedProgressiveTrainer
from UncertainGuidePGU.UG_unet import UncertaintyGuidedLoss


class DummyDataset(Dataset):
    """Dummy dataset for demonstration purposes"""
    
    def __init__(self, num_samples=100, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image (3 channels, RGB)
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Generate random binary mask (1 channel)
        mask = torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
        
        return image, mask


def demonstrate_uncertainty_guided_training():
    """
    Demonstrate the uncertainty-guided progressive training
    """
    print("Uncertainty-Guided Progressive U-Net Demonstration")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy datasets
    train_dataset = DummyDataset(num_samples=50, image_size=256)
    val_dataset = DummyDataset(num_samples=20, image_size=256)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Initialize uncertainty-guided trainer
    trainer = UncertaintyGuidedProgressiveTrainer(
        in_channels=3,
        num_classes=1,  # Binary segmentation
        device=device,
        uncertainty_alpha=1.0  # Uncertainty weighting factor
    )
    
    # Modify training parameters for demonstration (shorter training)
    for stage in trainer.stage_configs:
        trainer.stage_configs[stage]['epochs_per_stage'] = 3  # Reduced for demo
    
    print("\nTraining Configuration:")
    print(f"Uncertainty alpha: {trainer.uncertainty_alpha}")
    for stage, config in trainer.stage_configs.items():
        print(f"Stage {stage}: {config}")
    
    # Start progressive training with uncertainty guidance
    save_dir = Path('./demo_uncertainty_weights')
    trainer.train_progressive(
        train_loader=train_loader,
        val_loader=val_loader,
        max_stages=4,
        save_dir=save_dir
    )
    
    print("\nDemonstration completed!")
    print(f"Model weights saved to: {save_dir}")
    
    return trainer


def demonstrate_uncertainty_map_generation():
    """
    Demonstrate uncertainty map generation between stages
    """
    print("\nUncertainty Map Generation Demonstration")
    print("-" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample models and data
    from UncertainGuidePGU.UG_unet import PGUNet1, PGUNet2
    
    model_32x32 = PGUNet1(in_channels=3, num_classes=1).to(device)
    model_64x64 = PGUNet2(in_channels=3, num_classes=1).to(device)
    
    # Sample input (batch_size=2, channels=3, height=64, width=64)
    input_64x64 = torch.randn(2, 3, 64, 64).to(device)
    
    # Initialize uncertainty loss
    uncertainty_loss = UncertaintyGuidedLoss(device)
    
    print("Generating uncertainty map...")
    print(f"Input shape: {input_64x64.shape}")
    
    # Generate uncertainty map
    model_32x32.eval()
    uncertainty_map = uncertainty_loss.generate_uncertainty_map(
        input_current=input_64x64,
        model_prev=model_32x32,
        prev_resolution=32,
        current_resolution=64
    )
    
    print(f"Uncertainty map shape: {uncertainty_map.shape}")
    print(f"Uncertainty range: [{uncertainty_map.min():.4f}, {uncertainty_map.max():.4f}]")
    print(f"Uncertainty mean: {uncertainty_map.mean():.4f}")
    print(f"Uncertainty std: {uncertainty_map.std():.4f}")
    
    # Demonstrate loss weighting
    print("\nDemonstrating loss weighting...")
    
    # Create dummy target and loss function
    target = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    # Forward pass through 64x64 model
    model_64x64.eval()
    with torch.no_grad():
        output = model_64x64(input_64x64)
    
    # Apply uncertainty-weighted loss
    final_loss, base_loss = uncertainty_loss.apply_uncertainty_weighted_loss(
        loss_fn=loss_fn,
        output_current=output,
        target_current=target,
        uncertainty_map=uncertainty_map,
        alpha=1.0
    )
    
    print(f"Base loss (unweighted): {base_loss:.4f}")
    print(f"Final loss (weighted): {final_loss:.4f}")
    print(f"Loss difference: {final_loss - base_loss:.4f}")
    
    return uncertainty_map


def analyze_uncertainty_impact():
    """
    Analyze the impact of different uncertainty weighting factors
    """
    print("\nUncertainty Weighting Impact Analysis")
    print("-" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different alpha values
    alpha_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    # Create sample data
    from UncertainGuidePGU.UG_unet import PGUNet1
    model_prev = PGUNet1(in_channels=3, num_classes=1).to(device)
    model_prev.eval()
    
    input_data = torch.randn(2, 3, 64, 64).to(device)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)
    
    # Generate uncertainty map
    uncertainty_loss = UncertaintyGuidedLoss(device)
    uncertainty_map = uncertainty_loss.generate_uncertainty_map(
        input_current=input_data,
        model_prev=model_prev,
        prev_resolution=32,
        current_resolution=64
    )
    
    # Create dummy output
    output = torch.randn(2, 1, 64, 64).to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    print("Alpha\tBase Loss\tWeighted Loss\tDifference")
    print("-" * 45)
    
    for alpha in alpha_values:
        final_loss, base_loss = uncertainty_loss.apply_uncertainty_weighted_loss(
            loss_fn=loss_fn,
            output_current=output,
            target_current=target,
            uncertainty_map=uncertainty_map,
            alpha=alpha
        )
        
        difference = final_loss - base_loss
        print(f"{alpha:.1f}\t{base_loss:.4f}\t\t{final_loss:.4f}\t\t{difference:+.4f}")


if __name__ == "__main__":
    # Run demonstrations
    print("Starting Uncertainty-Guided Progressive U-Net Demonstrations\n")
    
    # 1. Demonstrate uncertainty map generation
    uncertainty_map = demonstrate_uncertainty_map_generation()
    
    # 2. Analyze impact of uncertainty weighting
    analyze_uncertainty_impact()
    
    # 3. Run full training demonstration (commented out for quick testing)
    # trainer = demonstrate_uncertainty_guided_training()
    
    print("\nAll demonstrations completed successfully!")
    print("\nKey Implementation Features:")
    print("✓ Uncertainty map generation from previous phase models")
    print("✓ Uncertainty-weighted loss calculation")
    print("✓ Progressive training with uncertainty guidance")
    print("✓ Enhanced monitoring and visualization")
    print("✓ Configurable uncertainty weighting factor (alpha)")