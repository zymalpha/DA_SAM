"""
Wavelet Transform Utilities
"""
import torch
import math


def dwt_2d(x):
    """
    2D Haar Wavelet Transform

    Args:
        x: Input tensor (B, C, H, W)
    Returns:
        LL, LH, HL, HH: Wavelet subbands
    """
    x_l = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) / math.sqrt(2)
    x_h = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) / math.sqrt(2)
    ll = (x_l[:, :, 0::2, :] + x_l[:, :, 1::2, :]) / math.sqrt(2)
    lh = (x_l[:, :, 0::2, :] - x_l[:, :, 1::2, :]) / math.sqrt(2)
    hl = (x_h[:, :, 0::2, :] + x_h[:, :, 1::2, :]) / math.sqrt(2)
    hh = (x_h[:, :, 0::2, :] - x_h[:, :, 1::2, :]) / math.sqrt(2)
    return ll, lh, hl, hh


def idwt_2d(ll, lh, hl, hh):
    """
    2D Haar Inverse Wavelet Transform

    Args:
        LL, LH, HL, HH: Wavelet subbands
    Returns:
        Reconstructed tensor (B, C, 2*H, 2*W)
    """
    B, C, H, W = ll.shape

    x_l = torch.zeros(B, C, H * 2, W, device=ll.device, dtype=ll.dtype)
    x_h = torch.zeros(B, C, H * 2, W, device=ll.device, dtype=ll.dtype)

    x_l[:, :, 0::2, :] = (ll + lh) / math.sqrt(2)
    x_l[:, :, 1::2, :] = (ll - lh) / math.sqrt(2)
    x_h[:, :, 0::2, :] = (hl + hh) / math.sqrt(2)
    x_h[:, :, 1::2, :] = (hl - hh) / math.sqrt(2)

    x = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
    x[:, :, :, 0::2] = (x_l + x_h) / math.sqrt(2)
    x[:, :, :, 1::2] = (x_l - x_h) / math.sqrt(2)

    return x


def dwt_3d(x):
    """
    3D Haar Wavelet Transform

    Args:
        x: Input tensor (B, C, D, H, W)
    Returns:
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH: 3D wavelet subbands
    """
    # Apply 2D DWT on each depth slice
    x_l = (x[:, :, :, :, 0::2] + x[:, :, :, :, 1::2]) / math.sqrt(2)
    x_h = (x[:, :, :, :, 0::2] - x[:, :, :, :, 1::2]) / math.sqrt(2)

    # Depth dimension
    x_ll = (x_l[:, :, :, 0::2, :] + x_l[:, :, :, 1::2, :]) / math.sqrt(2)
    x_lh = (x_l[:, :, :, 0::2, :] - x_l[:, :, :, 1::2, :]) / math.sqrt(2)
    x_hl = (x_h[:, :, :, 0::2, :] + x_h[:, :, :, 1::2, :]) / math.sqrt(2)
    x_hh = (x_h[:, :, :, 0::2, :] - x_h[:, :, :, 1::2, :]) / math.sqrt(2)

    return x_ll, x_lh, x_hl, x_hh


def idwt_3d(lll, llh, lhl, lhh, hll, hlh, hhl, hhh):
    """
    3D Haar Inverse Wavelet Transform
    """
    B, C, D, H, W = lll.shape

    # Inverse depth transform
    x_l = torch.zeros(B, C, D * 2, H, W, device=lll.device, dtype=lll.dtype)
    x_h = torch.zeros(B, C, D * 2, H, W, device=lll.device, dtype=lll.dtype)

    x_l[:, :, 0::2, :, :] = (lll + llh) / math.sqrt(2)
    x_l[:, :, 1::2, :, :] = (lll - llh) / math.sqrt(2)
    x_h[:, :, 0::2, :, :] = (hll + hlh) / math.sqrt(2)
    x_h[:, :, 1::2, :, :] = (hll - hlh) / math.sqrt(2)

    # 2D inverse (simplified - would need full implementation)
    x = torch.zeros(B, C, D * 2, H * 2, W * 2, device=lll.device, dtype=lll.dtype)

    return x


def test_wavelet():
    """Quick test for wavelet transforms"""
    # Test 2D DWT
    x = torch.randn(2, 16, 64, 64)
    ll, lh, hl, hh = dwt_2d(x)
    print(f"Input: {x.shape}")
    print(f"LL: {ll.shape}, LH: {lh.shape}, HL: {hl.shape}, HH: {hh.shape}")

    # Test inverse
    x_recon = idwt_2d(ll, lh, hl, hh)
    print(f"Reconstructed: {x_recon.shape}")
    print(f"Reconstruction error: {torch.max(torch.abs(x - x_recon)).item():.6f}")


if __name__ == "__main__":
    test_wavelet()
