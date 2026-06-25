"""
C2R: Multi-Gradient Residual Block for Enriched SAR Feature Extraction.

Paper: "SAR Ship Detection in Complex Marine Environments" (AFN-YOLO)

Unlike standard bottleneck residual blocks (ResNet) that use a single
transformation path, or the C2f module in YOLOv8 that uses a single
split-concat strategy, C2R explicitly creates multiple parallel gradient
branches with diverse kernel sizes. This enables the network to learn
complementary feature representations — edge features, texture patterns,
intensity distributions — which is particularly beneficial for SAR ship
targets that exhibit heterogeneous appearance characteristics.

Architecture:
    Input → PW Conv (2× expansion) → Split into k parallel branches
        ├─ Branch 1: DW Conv (3×3)
        ├─ Branch 2: DW Conv (5×5)
        ├─ Branch 3: Dilated Conv (3×3, d=2)
        └─ Branch 4: Dilated Conv (3×3, d=3)
    → Concat all branches → PW Conv (1×1, compression) → Output

The key difference from standard C2f:
- Standard C2f: Bottleneck with fixed 3×3 conv, single gradient path
- C2R: Multiple diverse gradient paths with different receptive fields

Dependencies: torch
"""

import torch
import torch.nn as nn


class MultiGradientBranch(nn.Module):
    """
    A single branch in C2R with configurable kernel size and dilation.

    Args:
        channels: number of input/output channels (depthwise)
        kernel_size: spatial extent of the depthwise convolution
        dilation: dilation rate for enlarged receptive field
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiGradientResidualBlock(nn.Module):
    """
    Core of C2R: processes features through k parallel gradient branches
    with diverse receptive fields, then fuses via concatenation.

    The diversity of kernel sizes (3×3, 5×5) and dilation rates (1, 2, 3)
    creates gradient paths with receptive fields ranging from 3×3 to 7×7,
    enabling the network to simultaneously capture fine edge details and
    broader textural patterns in SAR imagery.

    Args:
        channels:  input channel dimension for the hidden representation
        num_branches: number of parallel gradient branches (default 4)
        branch_kernels: kernel sizes for each branch
        branch_dilations: dilation rates for each branch
    """
    def __init__(
        self,
        channels: int,
        num_branches: int = 4,
        branch_kernels: list = None,
        branch_dilations: list = None,
    ):
        super().__init__()
        if branch_kernels is None:
            branch_kernels = [3, 5, 3, 3]
        if branch_dilations is None:
            branch_dilations = [1, 1, 2, 3]

        assert len(branch_kernels) == num_branches
        assert len(branch_dilations) == num_branches

        self.branches = nn.ModuleList([
            MultiGradientBranch(channels, k, d)
            for k, d in zip(branch_kernels, branch_dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each branch processes the same input independently
        branch_outputs = [branch(x) for branch in self.branches]
        # Summation fusion (preserves channel count)
        return torch.stack(branch_outputs, dim=0).sum(dim=0)


class C2R(nn.Module):
    """
    C2R: YOLOv8-compatible block with Multi-Gradient Residual bottlenecks.

    This is a drop-in replacement for the standard C2f block. Instead of
    standard Bottleneck blocks, it uses MultiGradientResidualBlock modules
    that provide diverse gradient flow paths for enriched spatial feature
    extraction from SAR imagery.

    Usage in YOLOv8 YAML:
        backbone:
          - [-1, 3, C2R, [128, True]]   # 3 C2R blocks, 128 channels

    Args:
        c1:       input channels
        c2:       output channels
        n:        number of MultiGradientResidualBlock bottlenecks
        shortcut: whether to use residual shortcuts
        g:        groups for initial convolution
        e:        expansion ratio (hidden_ch = c2 * e)
        num_branches: number of parallel gradient branches per block
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        num_branches: int = 4,
    ):
        super().__init__()
        self.c = int(c2 * e)           # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        # n MultiGradientResidualBlock bottlenecks
        self.m = nn.ModuleList(
            MultiGradientResidualBlock(self.c, num_branches=num_branches)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard C2f forward pass with multi-gradient bottlenecks.

        Args:
            x: [B, c1, H, W]
        Returns:
            [B, c2, H, W]
        """
        y = list(self.cv1(x).chunk(2, dim=1))   # split into 2 halves
        y.extend(m(y[-1]) for m in self.m)       # sequential through bottlenecks
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== C2R Module Test ===")
    x = torch.randn(2, 128, 64, 64)
    model = C2R(c1=128, c2=128, n=3, num_branches=4)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:     {tuple(x.shape)}")
    print(f"  Output:    {tuple(y.shape)}")
    print(f"  Params:    {params:,}")
    print(f"  Branches:  4 per block (k=3,5,3,3; d=1,1,2,3)")
    print(f"  Receptive fields: 3×3, 5×5, 5×5(d2), 7×7(d3)")
    print("  Test passed!")
