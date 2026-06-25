"""
AFN-YOLO: Adaptive Frequency Fusion Network for SAR Ship Detection.

Paper: "SAR Ship Detection in Complex Marine Environments"
Journal: Remote Sensing (under review)

Modules:
    c2f_ff   - C2f_FFCM: adaptive frequency-domain + spatial fusion
    c2r      - C2R: multi-gradient residual block
    tasfpn   - TripletAttention: cross-dimensional attention
"""

from .c2f_ff import C2f_FFCM, FourierUnit, Freq_Fusion, Fused_Fourier_Conv_Mixer
from .c2r import C2R, MultiGradientResidualBlock
from .tasfpn import TripletAttention, ZPool, BasicConv, AttentionGate

__all__ = [
    # C2f_FF
    'C2f_FFCM', 'FourierUnit', 'Freq_Fusion', 'Fused_Fourier_Conv_Mixer',
    # C2R
    'C2R', 'MultiGradientResidualBlock',
    # TAS-FPN
    'TripletAttention', 'ZPool', 'BasicConv', 'AttentionGate',
]
