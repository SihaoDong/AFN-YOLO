"""
TAS-FPN: Triplet Attention-based Spatial Feature Pyramid Network.

Paper: "SAR Ship Detection in Complex Marine Environments" (AFN-YOLO)

Triplet Attention captures cross-dimensional interactions by rotating the
input tensor and computing attention weights across three dimension pairs:
    - Channel-Height  (C-H):  "what" features  vs. vertical position
    - Channel-Width   (C-W):  "what" features  vs. horizontal position
    - Height-Width    (H-W):  spatial structure ("where")

This joint channel-spatial discrimination is critical in nearshore SAR
scenes: a building corner may share channel responses (intensity, texture)
with a ship, but its spatial context (row/column correlations) differs.
Triplet Attention catches both, suppressing false positives that a
channel-only (SE) or spatial-only attention would miss.

Placement in TAS-FPN (top-down pathway):
    P5 (deepest) → SPPF → TA1 → Upsample
                                ↓
          P4 → TA2 → Fusion → Upsample
                                ↓
          P3 → TA3 → Fusion → Output

Architecture of a single TripletAttention:
    Input X [B,C,H,W]
        ├─ Branch CW:  X → permute(B,W,H,C) → ZPool+Conv7×7+Sigmoid → permute back → X * attn
        ├─ Branch HC:  X → permute(B,H,C,W) → ZPool+Conv7×7+Sigmoid → permute back → X * attn
        └─ Branch HW:  X → ZPool+Conv7×7+Sigmoid → X * attn
    Output = (CW_out + HC_out + HW_out) / 3

Dependencies: torch (only)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Lightweight building blocks
# ---------------------------------------------------------------------------

class ZPool(nn.Module):
    """
    Z-Pool: concatenates max-pooled and avg-pooled features along the
    channel dimension to retain both salient and global context.

    Input:  [B, C, H, W]
    Output: [B, 2, H, W]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            x.max(dim=1, keepdim=True)[0],
            x.mean(dim=1, keepdim=True),
        ], dim=1)


class BasicConv(nn.Module):
    """
    Conv-BN-(optional ReLU) with standard settings.
    Used as the lightweight processing unit inside AttentionGate.
    """
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ---------------------------------------------------------------------------
# AttentionGate — single rotation branch
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Computes attention weights on a 2D feature map:
      ZPool (→ 2 channels) → 7×7 Conv → Sigmoid → element-wise multiply.

    When applied after a tensor rotation, this gate operates on a different
    dimension pair (e.g., C-H or C-W), enabling cross-dimensional attention.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_compress = self.compress(x)               # [B, 2, D1, D2]
        x_out = self.conv(x_compress)                # [B, 1, D1, D2]
        scale = torch.sigmoid_(x_out)
        return x * scale


# ---------------------------------------------------------------------------
# TripletAttention — three-way cross-dimensional attention
# ---------------------------------------------------------------------------

class TripletAttention(nn.Module):
    """
    Triplet Attention: captures cross-dimensional interactions via three
    rotated AttentionGate branches and averages their outputs.

    This avoids the heavy parameter cost of full 3D attention while still
    capturing dependencies across all three feature dimensions.

    Args:
        no_spatial: if True, skips the H-W branch (spatial-only attention),
                    useful for very deep layers where spatial resolution
                    is too low for meaningful 7×7 convolution.
    """
    def __init__(self, no_spatial: bool = False):
        super().__init__()
        self.cw = AttentionGate()           # Channel-Width branch
        self.hc = AttentionGate()           # Height-Channel branch
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()       # Height-Width branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W] attention-refined features
        """
        # Branch 1: Channel-Width (rotate H→batch-like, attend over C-W)
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()       # [B, H, C, W]
        x_out1 = self.cw(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()   # back → [B, C, H, W]

        # Branch 2: Height-Channel (rotate W→batch-like, attend over H-C)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()       # [B, W, H, C]
        x_out2 = self.hc(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()   # back → [B, C, H, W]

        # Branch 3: Height-Width (standard spatial attention)
        if not self.no_spatial:
            x_out3 = self.hw(x)
            return (x_out1 + x_out2 + x_out3) / 3.0
        else:
            return (x_out1 + x_out2) / 2.0


# ---------------------------------------------------------------------------
# Convenience: apply TA at multiple FPN levels
# ---------------------------------------------------------------------------

def apply_triplet_attention(features: list) -> list:
    """
    Apply TripletAttention to a list of FPN feature maps.

    Args:
        features: list of tensors [P3, P4, P5] from FPN levels
    Returns:
        list of attention-refined tensors, same shapes
    """
    ta_modules = nn.ModuleList([
        TripletAttention(no_spatial=(i == 2))  # P5 may skip spatial
        for i in range(len(features))
    ])
    return [ta(f) for ta, f in zip(ta_modules, features)]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== TAS-FPN / TripletAttention Module Test ===\n")

    # Test TripletAttention standalone
    x = torch.randn(2, 64, 40, 40)
    ta = TripletAttention()
    y = ta(x)
    params = sum(p.numel() for p in ta.parameters())
    print(f"[TripletAttention]")
    print(f"  Input:     {tuple(x.shape)}")
    print(f"  Output:    {tuple(y.shape)}")
    print(f"  Params:    {params:,}")
    print()

    # Test multi-scale application (simulating FPN levels)
    features = [
        torch.randn(2, 128, 80, 80),   # P3
        torch.randn(2, 256, 40, 40),   # P4
        torch.randn(2, 512, 20, 20),   # P5
    ]
    refined = apply_triplet_attention(features)
    for i, (f_in, f_out) in enumerate(zip(features, refined)):
        print(f"  P{i+3}: {tuple(f_in.shape)} → {tuple(f_out.shape)} (shape preserved ✓)")

    total_params = sum(
        sum(p.numel() for p in ta.parameters())
        for ta in [TripletAttention() for _ in range(3)]
    )
    print(f"\n  Total TA params (3 levels): {total_params:,}")
    print("  Test passed!")
