"""
Adapter modules for DA-SAM:
1. AIA - Adaptive Input Adapter
2. WFA (Wavelet Frequency Adapter) - Frequency-domain adapter for SAM encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings


class ConvBlock(nn.Module):
    """Basic convolutional block with GroupNorm and ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AIA(nn.Module):
    """
    Adaptive Input Adapter.
    Generates a modulation delta based on original and equalized images.
    """

    def __init__(self, in_ch=2, base_ch=32, mod_scale=0.5):
        super().__init__()
        initial_scale = torch.tensor(float(mod_scale))
        if not torch.isfinite(initial_scale).item():
            raise ValueError("AIA mod_scale must be finite")
        self.mod_scale = nn.Parameter(initial_scale)

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

        # Zero initialization for residual connection
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Older checkpoints used a fixed Python scalar, so they have no scale key.
        scale_key = prefix + "mod_scale"
        if scale_key not in state_dict:
            warnings.warn(
                "Legacy AIA checkpoint has no 'mod_scale'; using the initialized "
                f"value {self.mod_scale.detach().item():g} for the learnable scale.",
                UserWarning,
                stacklevel=3,
            )
            state_dict[scale_key] = self.mod_scale.detach().clone()
        else:
            scale = state_dict[scale_key]
            if (
                not torch.is_tensor(scale)
                or scale.shape != self.mod_scale.shape
                or scale.is_complex()
                or not torch.isfinite(scale).all().item()
            ):
                error_msgs.append(
                    f'Invalid AIA parameter "{scale_key}": expected a finite '
                    f'real scalar tensor with shape {tuple(self.mod_scale.shape)}.'
                )
                # PyTorch accepts some vector-to-scalar copies; prevent that
                # compatibility behavior from silently changing this parameter.
                state_dict[scale_key] = self.mod_scale.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, img_ori, img_eq):
        """
        Args:
            img_ori: Original image (B, 1, H, W)
            img_eq: Equalized image (B, 1, H, W)
        Returns:
            delta: Modulation (B, 1, H, W), bounded by abs(mod_scale)
        """
        x = torch.cat([img_ori, img_eq], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        # Bottleneck
        b = self.bottleneck(self.down2(e2))

        # Decoder
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Output modulation delta
        return torch.tanh(self.out_conv(d1)) * self.mod_scale


# ========================== DWT Functions ==========================
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


class WFA(nn.Module):
    """
    Wavelet Frequency Adapter (WFA)

    A frequency-domain adapter that uses 2D wavelet transform
    to process high and low frequency components separately.

    Key features:
    - Shared convolution for high-frequency subbands (LH, HL, HH)
    - Separate 1x1 conv for LL (low-frequency) subband
    - Zero-initialized residual connection for stable training
    """

    def __init__(self, dim, adapter_dim=32, scale=0.5):
        super().__init__()
        self.scale = scale
        self.adapter_dim = adapter_dim

        # Projection layers
        self.down_proj = nn.Linear(dim, adapter_dim)

        # LL subband: 1x1 conv (preserve low-frequency structure)
        self.ll_proj = nn.Conv2d(adapter_dim, adapter_dim, 1)

        # Shared conv for high-frequency subbands (LH, HL, HH)
        # This reduces parameters while maintaining effectiveness
        self.shared_conv = nn.Conv2d(adapter_dim, adapter_dim, 3, padding=1)

        self.up_proj = nn.Linear(adapter_dim, dim)

        # Zero initialization for residual connection
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, H, W, delta=None):
        """
        Args:
            x: Input tensor (B, N, C) where N = H * W
            H, W: Spatial dimensions
            delta: Unused (kept for API compatibility)
        Returns:
            Output tensor with residual connection (B, N, C)
        """
        B, N, C = x.shape

        # Project to adapter dimension
        h = self.down_proj(x)
        h_2d = h.transpose(1, 2).view(B, self.adapter_dim, H, W)

        # Padding for even dimensions (required for DWT)
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            h_2d = F.pad(h_2d, (0, pad_w, 0, pad_h), mode='reflect')

        # 2D Discrete Wavelet Transform
        ll, lh, hl, hh = dwt_2d(h_2d)

        # Process subbands
        # LL: 1x1 projection (preserves low-frequency content)
        ll_out = self.ll_proj(ll)

        # LH, HL, HH: shared 3x3 convolution (efficient high-freq processing)
        lh_out = self.shared_conv(lh)
        hl_out = self.shared_conv(hl)
        hh_out = self.shared_conv(hh)

        # Inverse DWT to reconstruct
        h_recon = idwt_2d(ll_out, lh_out, hl_out, hh_out)

        # Remove padding
        if pad_h or pad_w:
            h_recon = h_recon[:, :, :H, :W]

        # Reshape and project back
        h_recon = h_recon.view(B, self.adapter_dim, -1).transpose(1, 2)
        out = self.up_proj(h_recon)

        # Residual connection with scaling
        return x + out * self.scale


def test_adapters():
    """Quick test for adapter modules"""
    # Test AIA
    img_adapter = AIA(in_ch=2, base_ch=32, mod_scale=0.5)
    img_ori = torch.randn(2, 1, 256, 256)
    img_eq = torch.randn(2, 1, 256, 256)
    delta = img_adapter(img_ori, img_eq)
    print(f"AIA output shape: {delta.shape}")

    # Test WFA
    lwa = WFA(dim=768, adapter_dim=32, scale=0.5)
    x = torch.randn(2, 256, 768)  # (B, N, C)
    out = lwa(x, 16, 16)
    print(f"WFA output shape: {out.shape}")

    params_img = sum(p.numel() for p in img_adapter.parameters())
    params_lwa = sum(p.numel() for p in lwa.parameters())
    print(f"AIA params: {params_img / 1e3:.1f}K")
    print(f"WFA params: {params_lwa / 1e3:.1f}K")


if __name__ == "__main__":
    test_adapters()
