import torch
import torch.nn as nn


class BandProjectorMLP(nn.Module):
    """Pixel-wise 3-layer MLP that projects C_in bands → out_chans (default 6) bands.

    Implemented as three 1×1 convolutions so the pretrained patch_embed that follows
    receives a spatial map of the same H×W with exactly `out_chans` channels.
    """

    def __init__(self, in_chans: int, out_chans: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans,   hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_chans,  kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, C_in, H, W) → (B, out_chans, H, W)
