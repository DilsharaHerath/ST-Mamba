import torch
import torch.nn as nn
from mambaVision.models.mamba_vision import mamba_vision_B
from .temporal import PyramidTCN
from .fusion import LadderFusion

class MambaLadder(nn.Module):
    
    """
    SolarMamba: Multi-stage visual backbone + temporal PyramidTCN +
    temporal-conditioned fusion + multi-scale pooled prediction head.

    Visual Path:
        MambaVision-B produces four hierarchical feature maps.
        Hooks capture stage outputs without modifying the backbone.

    Temporal Path:
        PyramidTCN ingests a weather sequence and returns four temporal
        embeddings, one for each visual scale.

    Fusion:
        Each visual stage is modulated by a temporal gate via LadderFusion.
        No concatenation or cross-stage skip connections are used.

    Head:
        Global pooled features from all fused stages are concatenated and
        passed through an MLP to produce k* predictions for 1,5,10,15 minutes.
        
        UPDATE (Physics-Aware):
    Added a Direct Skip Connection for temporal embeddings to the final head.
    This solves the 'Image Blind' failure mode by allowing the model to act 
    as a pure time-series forecaster when visual features are weak or missing.
    """
    
    
    def __init__(self, pretrained=True, model_path=None):
        super().__init__()
        
        # 1. Visual Backbone (MambaVision-B)
        # MambaVision-B produces a sequence of feature maps of increasing depth
        
        if model_path:
            self.visual_backbone = mamba_vision_B(pretrained=pretrained, model_path=model_path)
        else:
            self.visual_backbone = mamba_vision_B(pretrained=pretrained)
        
        # Freeze visual backbone parameters
        # for param in self.visual_backbone.parameters():
        #     param.requires_grad = False
        
        # Store intermediate stage outputs via forward hooks
        self.features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        # Register hooks on the levels (stages)
        # MambaVision has self.levels which is a ModuleList
        for i, level in enumerate(self.visual_backbone.levels):
            level.register_forward_hook(get_activation(f'stage_{i+1}'))
            
        # 2. Temporal Backbone
        # Outputs four temporal embeddings of size 128.
        self.temporal_backbone = PyramidTCN(input_channels=7, embedding_dim=128)
        
        # 3. Fusion Blocks
        # MambaVision-B actual output dims (after downsampling in each stage):
        # Stage 1: 256 (128 -> 256)
        # Stage 2: 512 (256 -> 512)
        # Stage 3: 1024 (512 -> 1024)
        # Stage 4: 1024 (1024 -> 1024, no downsample)
        self.fusion1 = LadderFusion(visual_channels=256, temporal_channels=128)
        self.fusion2 = LadderFusion(visual_channels=512, temporal_channels=128)
        self.fusion3 = LadderFusion(visual_channels=1024, temporal_channels=128)
        self.fusion4 = LadderFusion(visual_channels=1024, temporal_channels=128)
        
        # 4. Head
        # GlobalAvgPool -> Concat -> MLP ->  4 regression outputs.
        # Concat size = 256 + 512 + 1024 + 1024 = 2816
        # + Temporal Skip (128) = 2944
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        input_dim = 2816 + 128  # INCREASED DIMENSION due to skip connection
        
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_dim), # Added Normalization
            nn.Dropout(0.5),    # Reduced Dropout
            nn.Linear(input_dim, 256),
            nn.GELU(),          # Switched to GELU
            nn.Dropout(0.5),    # Reduced Dropout
            nn.Linear(256, 4)   # Output k* for 4 horizons [1, 5, 10, 15]
        )

    def forward(self, image, weather_seq):
        # Clear previous features
        self.features = {}
        
        # Visual Forward
        _ = self.visual_backbone.forward_features(image)
        
        # Retrieve multi-stage visual features
        f1 = self.features['stage_1'] # (B, 128, H/4, W/4)
        f2 = self.features['stage_2'] # (B, 256, H/8, W/8)
        f3 = self.features['stage_3'] # (B, 512, H/16, W/16)
        f4 = self.features['stage_4'] # (B, 1024, H/32, W/32)
        
        # Temporal Forward
        # Returns a list of 4 temporal embeddings(All are size 128)
        t_feats = self.temporal_backbone(weather_seq) # [t1, t2, t3, t4]
        
        # Fusion
        out1 = self.fusion1(f1, t_feats[0])
        out2 = self.fusion2(f2, t_feats[1])
        out3 = self.fusion3(f3, t_feats[2])
        out4 = self.fusion4(f4, t_feats[3])
        
        # Pooling and Concat
        p1 = self.avg_pool(out1).flatten(1)
        p2 = self.avg_pool(out2).flatten(1)
        p3 = self.avg_pool(out3).flatten(1)
        p4 = self.avg_pool(out4).flatten(1)
        
        # --- PHYSICS-AWARE SKIP CONNECTION ---
        # We take the deepest temporal embedding (t_feats[3]) which contains
        # the most global context of the weather history.
        # This allows the MLP to see the raw weather trend even if 'Fusion' 
        # accidentally gates the visual features to zero.
        t_skip = t_feats[0]
        
        concat = torch.cat([p1, p2, p3, p4, t_skip], dim=1)
        
        # Prediction
        # Direct prediction (No residual connection)
        pred = self.head(concat)
        
        return pred
    
    