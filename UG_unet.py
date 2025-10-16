import torch
import torch.nn as nn
import torch.nn.functional as F

from UG_unet_parts import DoubleConv, DownSample, UpSample, InConv, Down, Up, OutConv


class UncertaintyGuidedLoss:
    """
    Uncertainty-guided loss weighting implementation for Progressive Growing U-Net
    
    This class provides functionality to generate uncertainty maps from previous
    phase models and apply uncertainty-weighted loss during training.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def generate_uncertainty_map(self, input_current, model_prev, prev_resolution, current_resolution):
        """
        Generate uncertainty map from previous phase model
        
        Args:
            input_current: Input image at current resolution (B, C, H, W)
            model_prev: Trained model from previous phase
            prev_resolution: Resolution of previous phase (int)
            current_resolution: Resolution of current phase (int)
        
        Returns:
            A_uncertainty: Uncertainty map at current resolution (B, 1, H, W)
        """
        model_prev.eval()
        
        with torch.no_grad():
            # Step a: Resize input down to previous phase resolution
            input_prev = F.interpolate(
                input_current, 
                size=(prev_resolution, prev_resolution), 
                mode='bilinear', 
                align_corners=True
            )
            
            # Step b: Pass through previous model to get probability output
            output_prev = model_prev(input_prev)
            P_prev = torch.sigmoid(output_prev)  # Convert logits to probabilities
            
            # Step c: Resize probability map back up to current resolution
            P_prev_upsampled = F.interpolate(
                P_prev, 
                size=(current_resolution, current_resolution), 
                mode='bilinear', 
                align_corners=True
            )
            
            # Step d: Calculate uncertainty map using the formula
            # A_uncertainty = 1.0 - 2.0 * |P_prev - 0.5|
            A_uncertainty = 1.0 - 2.0 * torch.abs(P_prev_upsampled - 0.5)
        
        return A_uncertainty.detach()  # Ensure no gradients
    
    def apply_uncertainty_weighted_loss(self, loss_fn, output_current, target_current, 
                                      uncertainty_map=None, alpha=1.0):
        """
        Apply uncertainty-weighted loss
        
        Args:
            loss_fn: Loss function (should have reduction='none')
            output_current: Model output at current resolution (B, C, H, W)
            target_current: Ground truth at current resolution (B, C, H, W)
            uncertainty_map: Uncertainty map (B, 1, H, W), None for stage 1
            alpha: Weight factor for uncertainty (default=1.0)
        
        Returns:
            final_loss: Weighted loss scalar
            pixel_loss: Unweighted pixel-wise loss for monitoring
        """
        # Calculate pixel-wise loss (reduction='none' to get per-pixel loss)
        pixel_loss = loss_fn(output_current, target_current)
        
        if uncertainty_map is None:
            # Stage 1: No uncertainty weighting
            final_loss = torch.mean(pixel_loss)
        else:
            # Stages 2+: Apply uncertainty weighting
            # Create weight map: weight = 1.0 + alpha * uncertainty
            weight_map = 1.0 + alpha * uncertainty_map
            
            # Apply weights element-wise
            weighted_loss_map = pixel_loss * weight_map.detach()
            
            # Calculate final loss as mean of weighted loss
            final_loss = torch.mean(weighted_loss_map)
        
        return final_loss, torch.mean(pixel_loss).item()


class UncertaintyGuidedProgressiveTrainer:
    """
    Enhanced Progressive Trainer with Uncertainty-Guided Loss Weighting
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.uncertainty_loss = UncertaintyGuidedLoss(device)
        self.stage_resolutions = {1: 32, 2: 64, 3: 128, 4: 256}
    
    def create_uncertainty_weighted_loss_fn(self, base_loss_fn):
        """
        Create a loss function that supports uncertainty weighting
        
        Args:
            base_loss_fn: Base loss function (e.g., BCEWithLogitsLoss)
        
        Returns:
            A loss function with reduction='none' for pixel-wise loss calculation
        """
        # Modify the base loss function to use reduction='none'
        if hasattr(base_loss_fn, 'reduction'):
            # Create a new instance with reduction='none'
            if isinstance(base_loss_fn, nn.BCEWithLogitsLoss):
                return nn.BCEWithLogitsLoss(
                    pos_weight=base_loss_fn.pos_weight,
                    reduction='none'
                )
            else:
                # For other loss functions, try to create with reduction='none'
                return type(base_loss_fn)(reduction='none')
        else:
            return base_loss_fn
    
    def uncertainty_guided_forward_pass(self, data, target, current_model, prev_model, 
                                      stage, loss_fn, alpha=1.0):
        """
        Perform forward pass with uncertainty-guided loss weighting
        
        Args:
            data: Input batch (B, C, H, W)
            target: Target batch (B, C, H, W) 
            current_model: Model for current stage
            prev_model: Model from previous stage (None for stage 1)
            stage: Current training stage (1, 2, 3, or 4)
            loss_fn: Loss function with reduction='none'
            alpha: Uncertainty weighting factor
        
        Returns:
            final_loss: Weighted loss for backpropagation
            metrics: Dictionary with loss metrics for monitoring
        """
        # Forward pass through current model
        output_current = current_model(data)
        
        # Generate uncertainty map if not stage 1
        uncertainty_map = None
        if stage > 1 and prev_model is not None:
            current_resolution = self.stage_resolutions[stage]
            prev_resolution = self.stage_resolutions[stage - 1]
            
            uncertainty_map = self.uncertainty_loss.generate_uncertainty_map(
                data, prev_model, prev_resolution, current_resolution
            )
        
        # Apply uncertainty-weighted loss
        final_loss, base_loss = self.uncertainty_loss.apply_uncertainty_weighted_loss(
            loss_fn, output_current, target, uncertainty_map, alpha
        )
        
        # Prepare metrics
        metrics = {
            'final_loss': final_loss.item(),
            'base_loss': base_loss,
            'uncertainty_weight_mean': torch.mean(uncertainty_map).item() if uncertainty_map is not None else 0.0,
            'uncertainty_weight_std': torch.std(uncertainty_map).item() if uncertainty_map is not None else 0.0
        }
        
        return final_loss, metrics


class PGUNet1(nn.Module):
    """Progressive Growing U-Net Stage 1 - 32x32 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.outc = OutConv(256, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # Initial conv, maintains resolution
        x2 = self.down4(x1)     # Downsample by 2
        x3 = self.up1(x2, x1)   # Upsample by 2, skip connection
        x = self.outc(x3)       # Final output - raw logits
        return x


class PGUNet2(nn.Module):
    """Progressive Growing U-Net Stage 2 - 64x64 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.outc1 = OutConv(256, num_classes)
        self.outc2 = OutConv(128, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # 64x64
        x2 = self.down3(x1)     # 32x32
        x3 = self.down4(x2)     # 16x16
        x4 = self.up1(x3, x2)   # 32x32
        x5 = self.up2(x4, x1)   # 64x64
        
        # Multi-scale outputs
        x4_out = self.outc1(x4)
        x5_out = self.outc2(x5)
        
        # Interpolate and combine
        x4_out = F.interpolate(x4_out, scale_factor=2, mode='bilinear', align_corners=True)
        x = x4_out + x5_out  # Raw logits for BCEWithLogitsLoss
        return x


class PGUNet3(nn.Module):
    """Progressive Growing U-Net Stage 3 - 128x128 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.outc1 = OutConv(256, num_classes)
        self.outc2 = OutConv(128, num_classes)
        self.outc3 = OutConv(64, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # 128x128
        x2 = self.down2(x1)     # 64x64
        x3 = self.down3(x2)     # 32x32
        x4 = self.down4(x3)     # 16x16
        x5 = self.up1(x4, x3)   # 32x32
        x6 = self.up2(x5, x2)   # 64x64
        x7 = self.up3(x6, x1)   # 128x128
        
        # Multi-scale outputs
        x5_out = self.outc1(x5)
        x6_out = self.outc2(x6)
        x7_out = self.outc3(x7)
        
        # Interpolate and combine
        x5_out = F.interpolate(x5_out, scale_factor=4, mode='bilinear', align_corners=True)
        x6_out = F.interpolate(x6_out, scale_factor=2, mode='bilinear', align_corners=True)
        x = x5_out + x6_out + x7_out  # Raw logits for BCEWithLogitsLoss
        return x


class PGUNet4(nn.Module):
    """Progressive Growing U-Net Stage 4 - 256x256 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc1 = OutConv(256, num_classes)
        self.outc2 = OutConv(128, num_classes)
        self.outc3 = OutConv(64, num_classes)
        self.outc4 = OutConv(64, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # 256x256
        x2 = self.down1(x1)     # 128x128
        x3 = self.down2(x2)     # 64x64
        x4 = self.down3(x3)     # 32x32
        x5 = self.down4(x4)     # 16x16
        x6 = self.up1(x5, x4)   # 32x32
        x7 = self.up2(x6, x3)   # 64x64
        x8 = self.up3(x7, x2)   # 128x128
        x9 = self.up4(x8, x1)   # 256x256
        
        # Multi-scale outputs
        x6_out = self.outc1(x6)
        x7_out = self.outc2(x7)
        x8_out = self.outc3(x8)
        x9_out = self.outc4(x9)
        
        # Interpolate and combine
        x6_out = F.interpolate(x6_out, scale_factor=8, mode='bilinear', align_corners=True)
        x7_out = F.interpolate(x7_out, scale_factor=4, mode='bilinear', align_corners=True)
        x8_out = F.interpolate(x8_out, scale_factor=2, mode='bilinear', align_corners=True)
        x = x6_out + x7_out + x8_out + x9_out  # Raw logits for BCEWithLogitsLoss
        return x


class ProgressiveUNet(nn.Module):
    """
    Progressive Growing U-Net with dynamic stage switching
    
    This implementation allows progressive training where the network starts
    with a simple architecture and progressively adds layers and increases
    resolution during training.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.current_stage = 1
        self.stage_resolutions = {1: 32, 2: 64, 3: 128, 4: 256}
        
        # Initialize all stages
        self.stage1 = PGUNet1(in_channels, num_classes)
        self.stage2 = PGUNet2(in_channels, num_classes)
        self.stage3 = PGUNet3(in_channels, num_classes)
        self.stage4 = PGUNet4(in_channels, num_classes)
        
        self.stages = {
            1: self.stage1,
            2: self.stage2,
            3: self.stage3,
            4: self.stage4
        }

    def set_stage(self, stage):
        """Set the current progressive stage (1-4)"""
        if stage not in [1, 2, 3, 4]:
            raise ValueError("Stage must be 1, 2, 3, or 4")
        self.current_stage = stage

    def get_current_resolution(self):
        """Get the target resolution for the current stage"""
        return self.stage_resolutions[self.current_stage]

    def transfer_weights(self, prev_stage_dict, current_stage_dict, stage):
        """
        Transfer weights from previous stage to current stage
        This is a simplified version - in practice, you'd need more sophisticated
        weight mapping based on layer correspondence
        """
        # A light-weight automatic transfer routine:
        # - If a key exists in both state_dicts and shapes match -> copy fully
        # - If shapes differ but both are tensors, attempt a partial copy on the
        #   leading dimensions (common practice for conv/linear weight copying)
        # - Otherwise leave the current value as initialized
        prev = prev_stage_dict
        cur = current_stage_dict
        new_state = {k: v.clone() for k, v in cur.items()}
        copied_keys = []

        for k, pv in prev.items():
            if k not in cur:
                continue
            cv = cur[k]
            # Only operate on tensors
            if not isinstance(pv, torch.Tensor) or not isinstance(cv, torch.Tensor):
                continue

            # Exact shape match -> copy
            if pv.shape == cv.shape:
                new_state[k] = pv.clone()
                copied_keys.append(k)
                continue

            # Partial copy heuristics for common tensor shapes
            try:
                if pv.ndim == 4 and cv.ndim == 4:
                    # Conv weight: (out, in, kH, kW)
                    out_c = min(pv.shape[0], cv.shape[0])
                    in_c = min(pv.shape[1], cv.shape[1])
                    tmp = cv.clone()
                    tmp[:out_c, :in_c, :, :] = pv[:out_c, :in_c, :, :]
                    new_state[k] = tmp
                    copied_keys.append(k)
                    continue

                if pv.ndim == 2 and cv.ndim == 2:
                    # Linear weight: (out, in)
                    out_c = min(pv.shape[0], cv.shape[0])
                    in_c = min(pv.shape[1], cv.shape[1])
                    tmp = cv.clone()
                    tmp[:out_c, :in_c] = pv[:out_c, :in_c]
                    new_state[k] = tmp
                    copied_keys.append(k)
                    continue

                if pv.ndim == 1 and cv.ndim == 1:
                    # Bias / BN running_*: (num_features,)
                    length = min(pv.shape[0], cv.shape[0])
                    tmp = cv.clone()
                    tmp[:length] = pv[:length]
                    new_state[k] = tmp
                    copied_keys.append(k)
                    continue
            except Exception:
                # If any unexpected shape issues occur, skip copying this key
                continue

        # Lightweight logging to help debugging when this is used interactively
        print(f"transfer_weights(stage={stage}): copied {len(copied_keys)} keys (examples: {copied_keys[:5]})")
        return new_state

    def forward(self, x, target_resolution=None):
        """
        Forward pass with optional resolution specification
        If target_resolution is provided, input will be resized accordingly
        """
        if target_resolution is not None:
            x = F.interpolate(x, size=(target_resolution, target_resolution), 
                            mode='bilinear', align_corners=True)
        else:
            target_resolution = self.get_current_resolution()
            x = F.interpolate(x, size=(target_resolution, target_resolution), 
                            mode='bilinear', align_corners=True)
        
        return self.stages[self.current_stage](x)


# Legacy UNet for compatibility
class UNet(nn.Module):
    """Original U-Net implementation for compatibility"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out