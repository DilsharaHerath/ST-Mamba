import torch
import torch.nn as nn
import torch.nn.functional as F
from mambaVision.models.mamba_vision import mamba_vision_B
from .temporal import PyramidTCN
from .fusion import LadderFusion


class MambaLadder(nn.Module):
    """
    Dual-mode:
      - Solar:  forward(image, weather_seq) -> (B,4)
      - TaxiBJ: forward(x_seq, ext_seq)     -> (B,H,2,32,32)

    TaxiBJ temporal input includes BOTH:
      - flow history embeddings
      - external features (time/holiday/meteorology)
    """

    def __init__(
        self,
        pretrained=True,
        model_path=None,
        task="solar",                   # "solar" or "taxibj"
        temporal_in_channels=7,          # Solar=7; TaxiBJ=(flow_embed_dim + E)
        taxibj_num_horizons=4,           # H
        flow_embed_dim=32,               # d_flow
        taxibj_out_h=32,
        taxibj_out_w=32,
    ):
        super().__init__()

        self.task = task
        self.taxibj_num_horizons = int(taxibj_num_horizons)
        self.flow_embed_dim = int(flow_embed_dim)
        self.taxibj_out_h = int(taxibj_out_h)
        self.taxibj_out_w = int(taxibj_out_w)

        # 1) Visual Backbone
        if model_path:
            self.visual_backbone = mamba_vision_B(pretrained=pretrained, model_path=model_path)
        else:
            self.visual_backbone = mamba_vision_B(pretrained=pretrained)

        self.features = {}

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        for i, level in enumerate(self.visual_backbone.levels):
            level.register_forward_hook(get_activation(f"stage_{i+1}"))

        # 2) Temporal Backbone
        self.temporal_backbone = PyramidTCN(input_channels=temporal_in_channels, embedding_dim=128)

        # 3) Fusion
        self.fusion1 = LadderFusion(visual_channels=256, temporal_channels=128)
        self.fusion2 = LadderFusion(visual_channels=512, temporal_channels=128)
        self.fusion3 = LadderFusion(visual_channels=1024, temporal_channels=128)
        self.fusion4 = LadderFusion(visual_channels=1024, temporal_channels=128)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ---- Solar head (unchanged behavior) ----
        solar_input_dim = 2816 + 128
        self.solar_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(solar_input_dim),
            nn.Dropout(0.5),
            nn.Linear(solar_input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

        # ---- TaxiBJ specific modules ----
        # flow (2ch) -> pseudo RGB for visual backbone
        self.flow2rgb = nn.Conv2d(2, 3, kernel_size=1)

        # flow history encoder for temporal branch: (B,T,2,32,32) -> (B,T,d_flow)
        self.flow_hist_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # (B,32,1,1)
            nn.Flatten(),             # (B,32)
            nn.Linear(32, self.flow_embed_dim),
            nn.GELU()
        )

        # TaxiBJ head: pooled concat -> (B, H*2*8*8) -> reshape -> upsample -> (B,H,2,32,32)
        taxibj_input_dim = 2816 + 128
        self.taxibj_head = nn.Sequential(
            nn.LayerNorm(taxibj_input_dim),
            nn.Linear(taxibj_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.taxibj_num_horizons * 2 * 8 * 8),
        )

    def _forward_backbone_and_fuse(self, image, temporal_seq):
        self.features = {}
        _ = self.visual_backbone.forward_features(image)

        f1 = self.features["stage_1"]
        f2 = self.features["stage_2"]
        f3 = self.features["stage_3"]
        f4 = self.features["stage_4"]

        t_feats = self.temporal_backbone(temporal_seq)  # [t1,t2,t3,t4], each (B,128)

        out1 = self.fusion1(f1, t_feats[0])
        out2 = self.fusion2(f2, t_feats[1])
        out3 = self.fusion3(f3, t_feats[2])
        out4 = self.fusion4(f4, t_feats[3])

        p1 = self.avg_pool(out1).flatten(1)
        p2 = self.avg_pool(out2).flatten(1)
        p3 = self.avg_pool(out3).flatten(1)
        p4 = self.avg_pool(out4).flatten(1)

        # keep the same skip behavior you had
        t_skip = t_feats[0]

        return p1, p2, p3, p4, t_skip

    def forward(self, a, b):
        """
        Solar:
          a=image (B,3,H,W), b=weather_seq (B,T,7) -> (B,4)

        TaxiBJ:
          a=x_seq (B,T,2,32,32), b=ext_seq (B,T,E) -> (B,H,2,32,32)
        """
        if self.task == "solar":
            image = a
            weather_seq = b
            p1, p2, p3, p4, t_skip = self._forward_backbone_and_fuse(image, weather_seq)
            concat = torch.cat([p1, p2, p3, p4, t_skip], dim=1)
            return self.solar_head(concat)

        if self.task == "taxibj":
            x_seq = a         # (B,T,2,32,32)
            ext_seq = b       # (B,T,E) or (B,T,0)

            B, T, C, H, W = x_seq.shape
            if C != 2:
                raise ValueError(f"TaxiBJ expects 2 channels (in/out), got {C}")

            # ----- Visual path: last flow frame -----
            x_last = x_seq[:, -1]                      # (B,2,32,32)
            image = self.flow2rgb(x_last)              # (B,3,32,32)
            image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)

            # ----- Temporal path: flow-history embedding + external features -----
            # Compute flow embedding per timestep (loop is fine at T<=12..24)
            flow_embs = []
            for t in range(T):
                emb_t = self.flow_hist_encoder(x_seq[:, t])  # (B,d_flow)
                flow_embs.append(emb_t)
            flow_emb = torch.stack(flow_embs, dim=1)         # (B,T,d_flow)

            if ext_seq is None or ext_seq.numel() == 0:
                temporal_seq = flow_emb
            else:
                temporal_seq = torch.cat([flow_emb, ext_seq], dim=-1)  # (B,T,d_flow+E)

            # ----- Shared fusion -----
            p1, p2, p3, p4, t_skip = self._forward_backbone_and_fuse(image, temporal_seq)
            concat = torch.cat([p1, p2, p3, p4, t_skip], dim=1)  # (B,2944)

            # ----- Spatial multi-horizon output -----
            z = self.taxibj_head(concat)  # (B, H*2*8*8)
            z = z.view(B, self.taxibj_num_horizons, 2, 8, 8)
            z = z.view(B * self.taxibj_num_horizons, 2, 8, 8)
            pred = F.interpolate(z, size=(self.taxibj_out_h, self.taxibj_out_w),
                                 mode="bilinear", align_corners=False)
            pred = pred.view(B, self.taxibj_num_horizons, 2, self.taxibj_out_h, self.taxibj_out_w)
            return pred

        raise ValueError(f"Unknown task: {self.task}")