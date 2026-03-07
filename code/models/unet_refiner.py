"""
U-Net Refiner for Stage 1: Coarse mask refinement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class UNetRefiner(nn.Module):
    """
    U-Net for coarse mask refinement.

    Input: Concatenation of (image, coarse_mask)
    Output: Refined segmentation mask
    """

    def __init__(self, in_ch=2, base_ch=32, depth=4):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))
            ch = out_ch

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, img, mask):
        """
        Args:
            img: Input image (B, 1, H, W)
            mask: Coarse mask (B, 1, H, W)
        Returns:
            Refined mask logits (B, 1, H, W)
        """
        x = torch.cat([img, mask], dim=1)

        # Encoder
        enc_feats = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            enc_feats.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = enc_feats[self.depth - 1 - i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)


def test_unet():
    """Quick test for U-Net"""
    model = UNetRefiner(in_ch=2, base_ch=32, depth=4)
    img = torch.randn(2, 1, 256, 256)
    mask = torch.randn(2, 1, 256, 256)
    out = model(img, mask)
    print(f"U-Net output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {params / 1e6:.2f}M")


if __name__ == "__main__":
    test_unet()
