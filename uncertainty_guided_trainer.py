import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from math import ceil
from pathlib import Path

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting functionality will be disabled.")

from UG_unet import (
    ProgressiveUNet, PGUNet1, PGUNet2, PGUNet3, PGUNet4, 
    UncertaintyGuidedLoss, UncertaintyGuidedProgressiveTrainer
)


class UncertaintyGuidedProgressiveTrainer:
    """
    Progressive Growing U-Net trainer with Uncertainty-Guided Loss Weighting
    
    This trainer implements:
    1. Progressive growing strategy with increasing resolution
    2. Uncertainty-guided loss weighting for phases N > 1
    3. Weight transfer between progressive stages
    """
    
    def __init__(self, in_channels=3, num_classes=1, device='cuda', uncertainty_alpha=1.0):
        self.device = device
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.uncertainty_alpha = uncertainty_alpha
        
        # Stage configurations
        self.stage_configs = {
            1: {'resolution': 32, 'epochs_per_stage': 40, 'lr': 3e-4},
            2: {'resolution': 64, 'epochs_per_stage': 40, 'lr': 1e-4},
            3: {'resolution': 128, 'epochs_per_stage': 40, 'lr': 1e-4},
            4: {'resolution': 256, 'epochs_per_stage': 40, 'lr': 1e-4}
        }
        
        # Initialize models for each stage
        self.models = {
            1: PGUNet1(in_channels, num_classes).to(device),
            2: PGUNet2(in_channels, num_classes).to(device),
            3: PGUNet3(in_channels, num_classes).to(device),
            4: PGUNet4(in_channels, num_classes).to(device)
        }
        
        self.current_stage = 1
        self.current_model = self.models[1]
        
        # Initialize uncertainty-guided loss
        self.uncertainty_loss = UncertaintyGuidedLoss(device)
        
        # Training components
        pos_weight = torch.tensor([5.0]).to(device)  # Weight positive class more
        self.base_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.optimizer = None
        self.setup_optimizer(1)
        
        # Training history with uncertainty metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'uncertainty_weights_mean': [],
            'uncertainty_weights_std': [],
            'base_loss': [],
            'stage_transitions': []
        }

    def setup_optimizer(self, stage):
        """Setup optimizer for the current stage"""
        lr = self.stage_configs[stage]['lr']
        self.optimizer = optim.RMSprop(
            self.current_model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )

    def dice_coefficient(self, pred, target, smooth=1):
        """Calculate Dice coefficient for binary segmentation evaluation"""
        pred = pred.to(target.device)
        target = target.to(pred.device)
        
        pred = pred.contiguous().float()
        target = target.contiguous().float()
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        
        dice = (2. * intersection + smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
        )
        
        return dice.mean()

    def get_predictions(self, output_batch):
        """Convert model output to predictions for binary segmentation"""
        probs = torch.sigmoid(output_batch)
        preds = (probs > 0.5).float()
        return preds.squeeze(1)

    def calculate_accuracy(self, pred, target):
        """Calculate pixel-wise accuracy"""
        pred = pred.to(target.device)
        assert pred.size() == target.size()
        bs, h, w = pred.size()
        n_pixels = bs * h * w
        incorrect = pred.ne(target).cpu().sum().numpy()
        err = incorrect / n_pixels
        return 1 - err

    def transfer_weights(self, prev_stage, new_stage):
        """
        Transfer weights from previous stage to new stage using the ProgressiveUNet method
        """
        print(f"Transferring weights from stage {prev_stage} to stage {new_stage}")
        
        # Get state dictionaries
        prev_dict = self.models[prev_stage].state_dict()
        new_dict = self.models[new_stage].state_dict()
        
        # Use the transfer_weights method from ProgressiveUNet
        progressive_unet = ProgressiveUNet(self.in_channels, self.num_classes)
        transferred_weights = progressive_unet.transfer_weights(prev_dict, new_dict, new_stage)
        
        # Load the transferred weights
        self.models[new_stage].load_state_dict(transferred_weights)
        print(f"Weight transfer completed for stage {new_stage}")

    def uncertainty_guided_forward_pass(self, data, target, stage):
        """
        Perform forward pass with uncertainty-guided loss weighting
        
        Args:
            data: Input batch (B, C, H, W)
            target: Target batch (B, C, H, W) 
            stage: Current training stage (1, 2, 3, or 4)
        
        Returns:
            final_loss: Weighted loss for backpropagation
            metrics: Dictionary with loss and uncertainty metrics
        """
        # Forward pass through current model
        output_current = self.current_model(data)
        
        # Generate uncertainty map if not stage 1
        uncertainty_map = None
        if stage > 1:
            prev_model = self.models[stage - 1]
            current_resolution = self.stage_configs[stage]['resolution']
            prev_resolution = self.stage_configs[stage - 1]['resolution']
            
            uncertainty_map = self.uncertainty_loss.generate_uncertainty_map(
                data, prev_model, prev_resolution, current_resolution
            )
        
        # Apply uncertainty-weighted loss
        final_loss, base_loss = self.uncertainty_loss.apply_uncertainty_weighted_loss(
            self.base_criterion, output_current, target, uncertainty_map, self.uncertainty_alpha
        )
        
        # Prepare metrics
        metrics = {
            'final_loss': final_loss.item(),
            'base_loss': base_loss,
            'output': output_current,
            'uncertainty_weight_mean': torch.mean(uncertainty_map).item() if uncertainty_map is not None else 0.0,
            'uncertainty_weight_std': torch.std(uncertainty_map).item() if uncertainty_map is not None else 0.0
        }
        
        return final_loss, metrics

    def train_epoch(self, dataloader, stage):
        """Train for one epoch with uncertainty-guided loss weighting"""
        self.current_model.train()
        
        # Set previous model to eval mode if exists
        if stage > 1:
            self.models[stage - 1].eval()
        
        total_loss = 0
        total_base_loss = 0
        total_dice = 0
        total_accuracy = 0
        total_uncertainty_mean = 0
        total_uncertainty_std = 0
        
        resolution = self.stage_configs[stage]['resolution']
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Resize inputs to current stage resolution
            data = F.interpolate(data, size=(resolution, resolution), mode='bilinear', align_corners=True)
            target = F.interpolate(target, size=(resolution, resolution), mode='nearest')
            
            self.optimizer.zero_grad()
            
            # Forward pass with uncertainty-guided loss
            final_loss, metrics = self.uncertainty_guided_forward_pass(data, target, stage)
            
            # Backward pass
            final_loss.backward()
            self.optimizer.step()
            
            # Calculate evaluation metrics
            output = metrics['output']
            pred = self.get_predictions(output)
            pred = pred.to(self.device)
            target_squeezed = target.squeeze(1)
            dice = self.dice_coefficient(pred.float(), target_squeezed.float())
            accuracy = self.calculate_accuracy(pred, target_squeezed.long())
            
            # Accumulate metrics
            total_loss += metrics['final_loss']
            total_base_loss += metrics['base_loss']
            total_dice += dice.item()
            total_accuracy += accuracy
            total_uncertainty_mean += metrics['uncertainty_weight_mean']
            total_uncertainty_std += metrics['uncertainty_weight_std']
            
            if batch_idx % 10 == 0:
                uncertainty_info = ""
                if stage > 1:
                    uncertainty_info = f", Unc_mean: {metrics['uncertainty_weight_mean']:.4f}"
                
                print(f'Stage {stage}, Batch {batch_idx}, Loss: {metrics["final_loss"]:.4f}, '
                      f'Base_Loss: {metrics["base_loss"]:.4f}, Dice: {dice.item():.4f}, '
                      f'Acc: {accuracy:.4f}{uncertainty_info}')
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_base_loss = total_base_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_uncertainty_mean = total_uncertainty_mean / num_batches
        avg_uncertainty_std = total_uncertainty_std / num_batches

        print(f"Stage {stage} training epoch completed. Batches processed: {num_batches}")
        
        return avg_loss, avg_base_loss, avg_dice, avg_accuracy, avg_uncertainty_mean, avg_uncertainty_std

    def validate_epoch(self, dataloader, stage):
        """Validate for one epoch"""
        self.current_model.eval()
        
        # Set previous model to eval mode if exists
        if stage > 1:
            self.models[stage - 1].eval()
        
        total_loss = 0
        total_base_loss = 0
        total_dice = 0
        total_accuracy = 0
        total_uncertainty_mean = 0
        total_uncertainty_std = 0
        
        resolution = self.stage_configs[stage]['resolution']
        
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Resize inputs to current stage resolution
                data = F.interpolate(data, size=(resolution, resolution), mode='bilinear', align_corners=True)
                target = F.interpolate(target, size=(resolution, resolution), mode='nearest')
                
                # Forward pass with uncertainty-guided loss
                final_loss, metrics = self.uncertainty_guided_forward_pass(data, target, stage)
                
                # Calculate evaluation metrics
                output = metrics['output']
                pred = self.get_predictions(output)
                pred = pred.to(self.device)
                target_squeezed = target.squeeze(1)
                dice = self.dice_coefficient(pred.float(), target_squeezed.float())
                accuracy = self.calculate_accuracy(pred, target_squeezed.long())
                
                # Accumulate metrics
                total_loss += metrics['final_loss']
                total_base_loss += metrics['base_loss']
                total_dice += dice.item()
                total_accuracy += accuracy
                total_uncertainty_mean += metrics['uncertainty_weight_mean']
                total_uncertainty_std += metrics['uncertainty_weight_std']
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_base_loss = total_base_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_uncertainty_mean = total_uncertainty_mean / num_batches
        avg_uncertainty_std = total_uncertainty_std / num_batches

        print(f"Stage {stage} validation epoch completed. Batches processed: {num_batches}")

        return avg_loss, avg_base_loss, avg_dice, avg_accuracy, avg_uncertainty_mean, avg_uncertainty_std

    def train_progressive(self, train_loader, val_loader, max_stages=4, save_dir='./uncertainty_guided_weights'):
        """
        Train the progressive U-Net through all stages with uncertainty-guided loss weighting
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print("Starting Uncertainty-Guided Progressive Growing U-Net Training")
        print("=" * 60)
        
        for stage in range(1, max_stages + 1):
            print(f"\nStarting Stage {stage}")
            print(f"Resolution: {self.stage_configs[stage]['resolution']}x{self.stage_configs[stage]['resolution']}")
            if stage > 1:
                print(f"Using uncertainty-guided loss weighting (alpha={self.uncertainty_alpha})")
            print("-" * 40)
            
            # Switch to new stage
            if stage > 1:
                self.transfer_weights(stage - 1, stage)
            
            self.current_stage = stage
            self.current_model = self.models[stage]
            self.setup_optimizer(stage)
            
            # Record stage transition
            self.history['stage_transitions'].append(len(self.history['train_loss']))
            
            # Training for this stage
            epochs = self.stage_configs[stage]['epochs_per_stage']
            best_val_dice = 0
            
            for epoch in range(epochs):
                start_time = time.time()
                
                # Train and validate with uncertainty-guided loss
                train_metrics = self.train_epoch(train_loader, stage)
                val_metrics = self.validate_epoch(val_loader, stage)
                
                (train_loss, train_base_loss, train_dice, train_acc, 
                 train_unc_mean, train_unc_std) = train_metrics
                 
                (val_loss, val_base_loss, val_dice, val_acc, 
                 val_unc_mean, val_unc_std) = val_metrics
                
                # Record history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_dice'].append(train_dice)
                self.history['val_dice'].append(val_dice)
                self.history['uncertainty_weights_mean'].append(val_unc_mean)
                self.history['uncertainty_weights_std'].append(val_unc_std)
                self.history['base_loss'].append(val_base_loss)
                
                epoch_time = time.time() - start_time
                
                print(f'Stage {stage}, Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)')
                print(f'Train - Loss: {train_loss:.4f}, Base: {train_base_loss:.4f}, '
                      f'Dice: {train_dice:.4f}, Acc: {train_acc:.4f}')
                print(f'Val   - Loss: {val_loss:.4f}, Base: {val_base_loss:.4f}, '
                      f'Dice: {val_dice:.4f}, Acc: {val_acc:.4f}')
                
                if stage > 1:
                    print(f'Uncertainty - Mean: {val_unc_mean:.4f}, Std: {val_unc_std:.4f}')
                
                # Save best model for this stage
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save({
                        'stage': stage,
                        'epoch': epoch,
                        'model_state_dict': self.current_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_dice': val_dice,
                        'train_dice': train_dice,
                        'uncertainty_alpha': self.uncertainty_alpha,
                        'history': self.history
                    }, save_path / f'ug_pgunet_stage{stage}_best.pth')
                
                print("-" * 60)
        
        print("Uncertainty-guided progressive training completed!")
        self.save_training_plots(save_path)

    def save_training_plots(self, save_path):
        """Save enhanced training history plots with uncertainty metrics"""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Skipping plot generation.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(len(self.history['train_loss']))
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss (Weighted)', alpha=0.7)
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss (Weighted)', alpha=0.7)
        ax1.plot(epochs, self.history['base_loss'], label='Base Loss (Unweighted)', alpha=0.7, linestyle='--')
        ax1.set_title('Loss Over Time (Uncertainty-Guided)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Add stage transition markers
        for transition in self.history['stage_transitions']:
            ax1.axvline(x=transition, color='red', linestyle='--', alpha=0.5)
        
        # Dice coefficient plot
        ax2.plot(epochs, self.history['train_dice'], label='Train Dice')
        ax2.plot(epochs, self.history['val_dice'], label='Val Dice')
        ax2.set_title('Dice Coefficient Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Coefficient')
        ax2.legend()
        
        # Add stage transition markers
        for transition in self.history['stage_transitions']:
            ax2.axvline(x=transition, color='red', linestyle='--', alpha=0.5)
        
        # Uncertainty weights plot
        ax3.plot(epochs, self.history['uncertainty_weights_mean'], label='Mean Uncertainty Weight')
        ax3.fill_between(epochs, 
                        np.array(self.history['uncertainty_weights_mean']) - np.array(self.history['uncertainty_weights_std']),
                        np.array(self.history['uncertainty_weights_mean']) + np.array(self.history['uncertainty_weights_std']),
                        alpha=0.3, label='±1 Std')
        ax3.set_title('Uncertainty Weights Over Time')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Uncertainty Weight')
        ax3.legend()
        
        # Add stage transition markers
        for transition in self.history['stage_transitions']:
            ax3.axvline(x=transition, color='red', linestyle='--', alpha=0.5)
        
        # Loss comparison plot (weighted vs unweighted)
        ax4.plot(epochs, np.array(self.history['val_loss']) - np.array(self.history['base_loss']), 
                label='Loss Difference (Weighted - Base)', alpha=0.7)
        ax4.set_title('Impact of Uncertainty Weighting')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add stage transition markers
        for transition in self.history['stage_transitions']:
            ax4.axvline(x=transition, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path / 'uncertainty_guided_training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_path / 'uncertainty_guided_training_plots.png'}")

    def load_stage_weights(self, stage, checkpoint_path):
        """Load weights for a specific stage"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.models[stage].load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights for stage {stage} from {checkpoint_path}")

    def save_uncertainty_analysis(self, data_loader, stage, save_path):
        """
        Save uncertainty analysis for a specific stage
        Generate uncertainty maps and statistics for analysis
        """
        if stage == 1:
            print("No uncertainty analysis for stage 1 (base stage)")
            return
        
        self.current_model.eval()
        self.models[stage - 1].eval()
        
        uncertainty_stats = []
        resolution = self.stage_configs[stage]['resolution']
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 10:  # Analyze first 10 batches
                    break
                
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Resize to current resolution
                data = F.interpolate(data, size=(resolution, resolution), mode='bilinear', align_corners=True)
                target = F.interpolate(target, size=(resolution, resolution), mode='nearest')
                
                # Generate uncertainty map
                prev_model = self.models[stage - 1]
                prev_resolution = self.stage_configs[stage - 1]['resolution']
                
                uncertainty_map = self.uncertainty_loss.generate_uncertainty_map(
                    data, prev_model, prev_resolution, resolution
                )
                
                # Calculate statistics
                stats = {
                    'batch_idx': batch_idx,
                    'uncertainty_mean': torch.mean(uncertainty_map).item(),
                    'uncertainty_std': torch.std(uncertainty_map).item(),
                    'uncertainty_min': torch.min(uncertainty_map).item(),
                    'uncertainty_max': torch.max(uncertainty_map).item()
                }
                uncertainty_stats.append(stats)
        
        # Save statistics
        import json
        with open(save_path / f'uncertainty_stats_stage{stage}.json', 'w') as f:
            json.dump(uncertainty_stats, f, indent=2)
        
        print(f"Uncertainty analysis saved for stage {stage}")