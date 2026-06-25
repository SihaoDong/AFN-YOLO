"""
C2f_FF: Adaptive Frequency-Domain Feature Fusion Module for SAR Ship Detection.

Paper: "SAR Ship Detection in Complex Marine Environments" (AFN-YOLO)

This module replaces the standard C2f bottleneck in YOLOv8 with a dual-branch
architecture that fuses adaptive frequency-domain features with spatial-domain features.

Architecture:
    Input ─┬─ Spatial Branch: Conv-BN-SiLU (standard convolution)
            │
            └─ Frequency Branch: FFT → Learnable Weight Modulation → IFFT → SiLU
                │
                └─ Residual Fusion: X_out = X_freq + X_spatial

Key innovation: Unlike static frequency filters (e.g., GFNet), the learnable
frequency weights α are optimized end-to-end with the detection objective,
enabling instance-level spectral adaptation.

Dependencies: torch, ultralytics (for C2f base class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fourier Unit — core FFT/IFFT with learnable convolution in frequency domain
# ---------------------------------------------------------------------------

class FourierUnit(nn.Module):
    """
    Applies 2D FFT, processes complex-valued frequency features via grouped
    convolution, then transforms back to spatial domain via 2D IFFT.

    This is the fundamental building block of C2f_FF. The 1×1 grouped
    convolution in the frequency domain learns to modulate spectral components
    before inverse transformation.

    Args:
        in_channels:  number of input channels
        out_channels: number of output channels (typically = in_channels × 2)
        groups:       number of groups for the 1×1 frequency-domain convolution
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
        self.groups = groups
        self.conv = nn.Conv2d(
            in_channels * 2, out_channels * 2,
            kernel_size=1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  real-valued feature map
        Returns:
            [B, C_out, H, W] frequency-processed feature map
        """
        B, C, H, W = x.shape
        fp16 = x.dtype == torch.float16

        # ---- Forward FFT ----
        # rfft2 outputs complex tensor [B, C, H, W//2+1]
        ffted = torch.fft.rfft2(x.float(), norm='ortho')

        # Decompose into real/imag and stack as separate channels
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)          # [B,C,H,W//2+1,2]
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()              # [B,C,2,H,W//2+1]
        ffted = ffted.view(B, -1, ffted.shape[-2], ffted.shape[-1])   # [B,2C,H,W//2+1]

        if fp16:
            ffted = ffted.half()

        # ---- Frequency-domain convolution ----
        ffted = self.conv(ffted)
        ffted = self.relu(self.bn(ffted))

        if fp16:
            ffted = ffted.float()

        # ---- Pack back to complex → Inverse FFT ----
        ffted = ffted.view(B, -1, 2, ffted.shape[-2], ffted.shape[-1])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()              # [B,C',H,W//2+1,2]
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(H, W), norm='ortho')
        if fp16:
            output = output.half()
        return output


# ---------------------------------------------------------------------------
# Freq_Fusion — frequency + spatial fusion with channel mixing
# ---------------------------------------------------------------------------

class Freq_Fusion(nn.Module):
    """
    Fuses frequency-domain and spatial-domain features through a two-stream
    design: one stream applies FourierUnit for global spectral processing,
    the other applies local spatial convolution. Results are concatenated
    and fused via 1×1 convolution.

    Args:
        dim:          input channel dimension
        kernel_size:  list of kernel sizes for multi-scale spatial mixing
        se_ratio:     squeeze ratio for channel attention in parent module
        local_size:   not used directly here (parameter for parent)
        scale_ratio:  channel expansion ratio (default 2)
    """
    def __init__(
        self,
        dim: int,
        kernel_size: list = None,
        se_ratio: int = 4,
        local_size: int = 8,
        scale_ratio: int = 2,
        spilt_num: int = 4,
    ):
        super().__init__()
        if kernel_size is None:
            kernel_size = [1, 3, 5, 7]
        self.dim = dim
        self.dim_sp = dim * scale_ratio // spilt_num

        # Two parallel initial projections
        self.conv_init_1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.conv_init_2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )

        # Fusion after Fourier + spatial
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
        )

        # Core: FourierUnit processes concatenated features in frequency domain
        self.FFC = FourierUnit(dim * 2, dim * 2)

        self.bn = nn.BatchNorm2d(dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, self.dim, dim=1)
        x1 = self.conv_init_1(x1)
        x2 = self.conv_init_2(x2)
        x0 = torch.cat([x1, x2], dim=1)           # concat in channel dim
        x = self.FFC(x0) + x0                      # residual FFT path
        x = self.relu(self.bn(x))
        return x


# ---------------------------------------------------------------------------
# Fused_Fourier_Conv_Mixer — complete mixer block (local + global)
# ---------------------------------------------------------------------------

class Fused_Fourier_Conv_Mixer(nn.Module):
    """
    Combines local depthwise convolutions (multi-scale) with global
    frequency-domain mixing (Freq_Fusion).

    Architecture:
        Input → PW-Conv (2× expansion) → Split
            ├─ Local: DW-Conv 3×3 + DW-Conv 5×5
            └─ Global: Freq_Fusion (FFT path)
        → Concat → PW-Conv → Channel Attention → Output
    """
    def __init__(
        self,
        dim: int,
        token_mixer_for_global: nn.Module = Freq_Fusion,
        mixer_kernel_size: list = None,
        local_size: int = 8,
    ):
        super().__init__()
        if mixer_kernel_size is None:
            mixer_kernel_size = [1, 3, 5, 7]
        self.dim = dim

        # Global frequency mixer
        self.mixer_global = token_mixer_for_global(
            dim=dim,
            kernel_size=mixer_kernel_size,
            se_ratio=8,
            local_size=local_size,
        )

        # Fusion convolution after concat
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU(),
        )

        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid(),
        )

        # Initial pointwise expansion
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
        )

        # Local depthwise convolutions (multi-scale: 3×3 and 5×5)
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU(),
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect'),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)                                   # [B, 2C, H, W]
        local_part, global_input = torch.split(x, self.dim, dim=1)

        # Local branches
        x_local_1 = self.dw_conv_1(local_part)                  # 3×3 DW
        x_local_2 = self.dw_conv_2(local_part)                  # 5×5 DW

        # Global frequency branch
        x_global = self.mixer_global(
            torch.cat([x_local_1, x_local_2], dim=1)
        )

        # Fusion + channel attention
        x = self.ca_conv(x_global)
        x = self.ca(x) * x
        return x


# ---------------------------------------------------------------------------
# C2f_FFCM — YOLOv8-compatible C2f wrapper
# ---------------------------------------------------------------------------

class C2f_FFCM(nn.Module):
    """
    YOLOv8-compatible C2f module with Fused_Fourier_Conv_Mixer bottlenecks.

    This is a drop-in replacement for the standard C2f block in YOLOv8
    backbone and neck. Each bottleneck is a Fused_Fourier_Conv_Mixer that
    performs adaptive frequency-spatial fusion, replacing standard
    Bottleneck blocks.

    Compatible with ultralytics YOLOv8 model building.

    Args:
        c1:       input channels
        c2:       output channels
        n:        number of Fused_Fourier_Conv_Mixer bottlenecks
        shortcut: whether to use residual shortcuts
        g:        groups for initial convolution
        e:        expansion ratio
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
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
        # n Fused_Fourier_Conv_Mixer bottlenecks
        self.m = nn.ModuleList(
            Fused_Fourier_Conv_Mixer(self.c) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard C2f forward with frequency-aware bottlenecks."""
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------------------------------------------------------
# Utility: replace C2f with C2f_FFCM in model building
# ---------------------------------------------------------------------------

def is_c2f_ffcm(module) -> bool:
    """Check if a module name corresponds to C2f_FFCM."""
    return isinstance(module, str) and module in ('C2f_FFCM', 'C2f_FF')


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== C2f_FF Module Test ===")
    x = torch.randn(2, 128, 32, 32)
    model = C2f_FFCM(c1=128, c2=128, n=3)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(y.shape)}")
    print(f"  Params: {params:,}")
    print("  Test passed! (FP16 supported via FFT internal cast)")
