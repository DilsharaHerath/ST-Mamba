import torch
import torch.nn as nn

class LadderFusion(nn.Module):
    """
    Temporal-conditioned channel gating for spatial visual features.

    This module injects temporal context into visual features by generating
    a channel-wise gate from temporal descriptors. The gate modulates the
    visual backbone's output through multiplicative scaling with a residual
    skip connection.

    Inputs
    ------
    visual : Tensor
        Shape (B, C_v, H, W). Spatial visual feature maps.
    temporal : Tensor
        Shape (B, C_t). Global pooled temporal representation (e.g., from PyramidTCN).

    Output
    ------
    Tensor of shape (B, C_v, H, W) where each visual channel is scaled based
    on temporal context:  visual * σ(W_t·temporal) + visual
    """
    
    def __init__(self, visual_channels, temporal_channels):
        super().__init__()
        self.project = nn.Linear(temporal_channels, visual_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual, temporal):
        
        # Project temporal vector to match visual channel dimension
        temp_proj = self.project(temporal) # (B, C_v)
        
        # Reshape for broadcasting: (B, C_v, 1, 1)
        temp_proj = temp_proj.view(temp_proj.shape[0], temp_proj.shape[1], 1, 1)
        
        # Sigmoid Gate: Visual_Out = Visual * Sigmoid(Temp_Proj) + Visual
        gate = self.sigmoid(temp_proj)
        
        # Residual gated modulation
        out = visual * gate + visual
        
        return out
