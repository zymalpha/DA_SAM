"""
DA-SAM: Dual-Adapter SAM for CBCT Projection-Domain Metal Trace Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_refiner import UNetRefiner
from .adapter import AIA, WFA


def mask_to_logits_256(mask, logit_scale=10.0):
    """Convert mask to SAM prompt format"""
    lowres = F.interpolate(mask, size=(256, 256), mode='nearest')
    return (lowres * 2. - 1.) * logit_scale


class DA_SAM(nn.Module):
    """
    Two-Stage Dual-Adapter SAM with Wavelet Frequency Adapter

    This model combines:
    1. Stage 1: U-Net Refiner for coarse mask refinement
    2. Stage 2: SAM with dual adapters (Image + WFA)
    """

    def __init__(self, sam_model, unet, aia, freq_adapters, cfg):
        super().__init__()
        self.sam = sam_model
        self.unet = unet
        self.aia = aia
        self.freq_adapters = freq_adapters
        self.cfg = cfg

        # SAM normalization constants
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1))

    def forward(self, img_ori, img_eq, mask_coarse):
        """
        Args:
            img_ori: Original projection image (B, 1, H, W)
            img_eq: Equalized projection image (B, 1, H, W)
            mask_coarse: Coarse metal trace mask (B, 1, H, W)
        Returns:
            pred_logits: Final prediction (B, 1, H, W)
            unet_logits: Stage 1 U-Net output (B, 1, H, W)
            delta: Image adapter modulation (B, 1, H, W)
        """
        B, _, H, W = img_ori.shape

        # ===== Stage 1: U-Net refinement =====
        unet_logits = self.unet(img_ori, mask_coarse)
        refined_mask = torch.sigmoid(unet_logits)
        mask_prompt = mask_to_logits_256(refined_mask, self.cfg.LOGIT_SCALE)

        # ===== Image Adapter =====
        delta = self.aia(img_ori, img_eq)
        img_adapted = (img_ori * (1 + delta)).clamp(0, 1)

        # ===== Prepare 3-channel input for SAM =====
        # Stack: [adapted, equalized, original]
        x_3ch = torch.cat([img_adapted, img_eq, img_ori], dim=1)
        x_3ch = (x_3ch * 255.0 - self.pixel_mean) / self.pixel_std

        # ===== SAM Encoder with WFA =====
        img_emb = self._forward_encoder(x_3ch, delta)

        # ===== SAM Decoder =====
        img_pe = self.sam.prompt_encoder.get_dense_pe()
        pred_list = []

        for i in range(B):
            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=None, boxes=None, masks=mask_prompt[i:i + 1]
            )
            low_res, _ = self.sam.mask_decoder(
                image_embeddings=img_emb[i:i + 1],
                image_pe=img_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False
            )
            pred_list.append(low_res)

        low_res_logits = torch.cat(pred_list, dim=0)
        pred_logits = F.interpolate(low_res_logits, size=(H, W), mode='bilinear', align_corners=False)

        return pred_logits, unet_logits, delta

    def _forward_encoder(self, x, delta):
        """
        Forward through SAM encoder with WFA adapters inserted

        Args:
            x: Input tensor (B, 3, H, W)
            delta: Unused (for API compatibility)
        Returns:
            Encoded image features
        """
        encoder = self.sam.image_encoder

        # Patch embedding
        x = encoder.patch_embed(x)
        if encoder.pos_embed is not None:
            x = x + encoder.pos_embed

        B, H, W, C = x.shape

        # Apply blocks with WFA adapters
        for i, block in enumerate(encoder.blocks):
            x = block(x)
            if i < len(self.freq_adapters):
                x = x.view(B, H * W, C)
                x = self.freq_adapters[i](x, H, W, delta)
                x = x.view(B, H, W, C)

        # Apply neck
        return encoder.neck(x.permute(0, 3, 1, 2))

    def freeze_stage1(self):
        """Freeze Stage 1 (U-Net) parameters"""
        for p in self.unet.parameters():
            p.requires_grad = False

    def freeze_sam_encoder(self):
        """Freeze SAM encoder parameters"""
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False

    def get_trainable_params(self):
        """Get trainable parameters for optimizer"""
        trainable_params = []
        trainable_params.append({'params': self.aia.parameters()})
        trainable_params.append({'params': self.freq_adapters.parameters()})
        trainable_params.append({'params': self.sam.mask_decoder.parameters()})
        return trainable_params


def build_da_sam(sam_model, cfg):
    """
    Build DA-SAM model

    Args:
        sam_model: Pre-loaded SAM model
        cfg: Configuration object
    Returns:
        model: DA-SAM model
    """
    # U-Net Refiner
    unet = UNetRefiner(in_ch=2, base_ch=cfg.UNET_BASE_CH, depth=cfg.UNET_DEPTH)

    # Image Adapter
    aia = AIA(in_ch=2, base_ch=cfg.AIA_CH, mod_scale=cfg.MOD_SCALE)

    # WFA adapters for each transformer block
    num_blocks = len(sam_model.image_encoder.blocks)
    embed_dim = sam_model.image_encoder.patch_embed.proj.out_channels
    freq_adapters = nn.ModuleList([
        WFA(embed_dim, cfg.ADAPTER_DIM, cfg.ADAPTER_SCALE) for _ in range(num_blocks)
    ])

    # Full model
    model = DA_SAM(sam_model, unet, aia, freq_adapters, cfg)

    return model


def test_da_sam():
    """Quick test for DA-SAM"""
    from segment_anything import sam_model_registry

    class SimpleCfg:
        UNET_BASE_CH = 32
        UNET_DEPTH = 4
        AIA_CH = 32
        MOD_SCALE = 0.5
        ADAPTER_DIM = 32
        ADAPTER_SCALE = 0.5
        LOGIT_SCALE = 10.0

    print("Testing DA-SAM components...")

    # Test U-Net
    unet = UNetRefiner(in_ch=2, base_ch=32, depth=4)
    print(f"U-Net params: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f}M")

    # Test adapters
    img_adapter = AIA(in_ch=2, base_ch=32, mod_scale=0.5)
    print(f"AIA params: {sum(p.numel() for p in img_adapter.parameters()) / 1e3:.1f}K")

    lwa = WFA(dim=768, adapter_dim=32, scale=0.5)
    print(f"WFA params: {sum(p.numel() for p in lwa.parameters()) / 1e3:.1f}K")

    print("All tests passed!")


if __name__ == "__main__":
    test_da_sam()

