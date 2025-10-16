#!/usr/bin/env python3
"""
Simple test script for uncertainty-guided loss implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UncertainGuidePGU.UG_unet import UncertaintyGuidedLoss, PGUNet1
import torch
import torch.nn as nn

def test_uncertainty_implementation():
    print('Testing uncertainty-guided loss implementation...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Test uncertainty map generation
    model_prev = PGUNet1(3, 1).to(device)
    uncertainty_loss = UncertaintyGuidedLoss(device)
    input_data = torch.randn(2, 3, 64, 64).to(device)

    model_prev.eval()
    uncertainty_map = uncertainty_loss.generate_uncertainty_map(input_data, model_prev, 32, 64)
    print(f'Uncertainty map shape: {uncertainty_map.shape}')
    print(f'Uncertainty range: [{uncertainty_map.min():.4f}, {uncertainty_map.max():.4f}]')
    print('✓ Uncertainty map generation successful!')

    # Test uncertainty-weighted loss
    target = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)
    output = torch.randn(2, 1, 64, 64).to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    final_loss, base_loss = uncertainty_loss.apply_uncertainty_weighted_loss(
        loss_fn, output, target, uncertainty_map, alpha=1.0
    )
    print(f'Base loss: {base_loss:.4f}, Weighted loss: {final_loss:.4f}')
    print('✓ Uncertainty-weighted loss calculation successful!')
    print('Implementation test completed successfully!')
    return True

if __name__ == "__main__":
    test_uncertainty_implementation()