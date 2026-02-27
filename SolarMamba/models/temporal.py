import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidTCN(nn.Module):
    def __init__(self, input_channels=7, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_channels, embedding_dim)
        
        # 4 Parallel branches with different dilations
        # We want to capture different temporal scales.
        # Assuming we pool the time dimension to get a single vector per branch
        # or we keep time and pool later. 
        # The fusion block expects (B, C_t). So we should probably pool over time here.
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: (B, T, C)
        x = self.embedding(x) # (B, T, Emb)
        x = x.permute(0, 2, 1) # (B, Emb, T)
        
        t1 = self.branch1(x).squeeze(-1) # (B, Emb)
        t2 = self.branch2(x).squeeze(-1) # (B, Emb)
        t3 = self.branch3(x).squeeze(-1) # (B, Emb)
        t4 = self.branch4(x).squeeze(-1) # (B, Emb)
        
        return [t1, t2, t3, t4]
