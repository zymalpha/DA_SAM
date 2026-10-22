"""
Dataset classes for Metal Artifact Reduction
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide


def resize_binary_mask(mask, target_hw):
    """Resize a 2D binary mask without creating interpolated foreground pixels."""
    height, width = target_hw
    if mask.ndim != 2 or height <= 0 or width <= 0:
        raise ValueError("Binary masks must be 2D and have a positive target size")
    return cv2.resize(
        (mask > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


class BaseMetalDataset(Dataset):
    """Base dataset class for metal artifact reduction"""

    def __init__(self, names, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.rsz = ResizeLongestSide(cfg.IMG_SIZE)

        self.samples = self._load_samples(names)

        print(f"[{split}] Loaded {len(self.samples)} samples from {len(names)} volumes")

    def _load_samples(self, names):
        """Load sample list from names"""
        samples = []
        for name in names:
            proj_dir = os.path.join(self.cfg.DIR_PROJ, name)
            if not os.path.exists(proj_dir):
                continue
            for f in sorted(os.listdir(proj_dir)):
                if f.endswith(".png"):
                    samples.append((name, f))
        return samples

    def _find_file(self, folder, suffix):
        """Find file with given suffix in folder"""
        if not os.path.exists(folder):
            return None
        for f in os.listdir(folder):
            if f.endswith(suffix):
                return os.path.join(folder, f)
        return None

    def _get_suffix(self, fname):
        """Extract suffix from filename"""
        return "_" + "_".join(fname.split("_")[2:])

    def _pad_image(self, img):
        """Pad image to square"""
        padded = np.zeros((self.cfg.IMG_SIZE, self.cfg.IMG_SIZE), dtype=np.uint8)
        padded[:img.shape[0], :img.shape[1]] = img
        return padded

    def _to_tensor(self, img, normalize=True):
        """Convert image to tensor"""
        if normalize:
            img = img / 255.0
        return torch.from_numpy(img).float().unsqueeze(0)

    def __len__(self):
        return len(self.samples)


class Stage1Dataset(BaseMetalDataset):
    """
    Dataset for Stage 1: U-Net training

    Loads:
    - Input: Original projection image + Metal trace mask
    - Output: Ground truth label
    """

    def __getitem__(self, idx):
        name, fname = self.samples[idx]
        suffix = self._get_suffix(fname)

        # Load images
        img = cv2.imread(os.path.join(self.cfg.DIR_PROJ, name, fname), cv2.IMREAD_GRAYSCALE)
        mask_path = self._find_file(os.path.join(self.cfg.DIR_PROJ_METAL, name), suffix)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else np.zeros_like(img)
        gt_path = self._find_file(os.path.join(self.cfg.DIR_LABEL, name), suffix)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing files
        if img is None or gt is None:
            return self.__getitem__((idx + 1) % len(self))

        # Resize
        img = self.rsz.apply_image(img)
        mask = resize_binary_mask(mask, img.shape[:2])
        gt = resize_binary_mask(gt, img.shape[:2])

        # Pad to square
        img = self._pad_image(img)
        mask = self._pad_image(mask)
        gt = self._pad_image(gt)

        # To tensor
        img_t = self._to_tensor(img)
        mask_t = self._to_tensor((mask > 0).astype(np.float32), normalize=False)
        gt_t = self._to_tensor((gt > 0).astype(np.float32), normalize=False)

        return {"img": img_t, "mask": mask_t, "gt": gt_t}


class MetalArtifactDataset(BaseMetalDataset):
    """
    Dataset for Stage 2: DA-SAM training

    Loads:
    - img_ori: Original projection image
    - img_eq: Equalized projection image
    - mask_coarse: Coarse metal trace mask (from Stage 1 or direct)
    - gt: Ground truth label
    """

    def __getitem__(self, idx):
        name, fname = self.samples[idx]
        suffix = self._get_suffix(fname)

        # Load images
        img_ori = cv2.imread(os.path.join(self.cfg.DIR_PROJ, name, fname), cv2.IMREAD_GRAYSCALE)
        img_eq_path = self._find_file(os.path.join(self.cfg.DIR_EQ, name), suffix)
        img_eq = cv2.imread(img_eq_path, cv2.IMREAD_GRAYSCALE) if img_eq_path else img_ori

        mask_path = self._find_file(os.path.join(self.cfg.DIR_PROJ_METAL, name), suffix)
        mask_coarse = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else np.zeros_like(img_ori)

        gt_path = self._find_file(os.path.join(self.cfg.DIR_LABEL, name), suffix)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing files
        if img_ori is None or mask_coarse is None or gt is None:
            return self.__getitem__((idx + 1) % len(self))

        # Resize
        img_ori = self.rsz.apply_image(img_ori)
        img_eq = self.rsz.apply_image(img_eq)
        mask_coarse = resize_binary_mask(mask_coarse, img_ori.shape[:2])
        gt = resize_binary_mask(gt, img_ori.shape[:2])

        # Pad to square
        img_ori = self._pad_image(img_ori)
        img_eq = self._pad_image(img_eq)
        mask_coarse = self._pad_image(mask_coarse)
        gt = self._pad_image(gt)

        # To tensor
        img_ori_t = self._to_tensor(img_ori)
        img_eq_t = self._to_tensor(img_eq)
        mask_t = self._to_tensor((mask_coarse > 0).astype(np.float32), normalize=False)
        gt_t = self._to_tensor((gt > 0).astype(np.float32), normalize=False)

        return {"img_ori": img_ori_t, "img_eq": img_eq_t, "mask_coarse": mask_t, "gt": gt_t}


class InferenceDataset(Dataset):
    """
    Dataset for inference/testing

    Returns image paths along with processed tensors for later saving
    """

    def __init__(self, names, cfg):
        self.cfg = cfg
        self.rsz = ResizeLongestSide(cfg.IMG_SIZE)

        self.samples = self._load_samples(names)
        print(f"[Inference] Loaded {len(self.samples)} samples")

    def _load_samples(self, names):
        """Load samples with file paths"""
        samples = []
        for name in names:
            proj_dir = os.path.join(self.cfg.DIR_PROJ, name)
            if not os.path.exists(proj_dir):
                continue
            for fname in sorted(os.listdir(proj_dir)):
                if not fname.endswith(".png"):
                    continue
                suffix = "_" + "_".join(fname.split("_")[2:])

                def find_file(folder):
                    if not os.path.exists(folder):
                        return None
                    for f in os.listdir(folder):
                        if f.endswith(suffix):
                            return os.path.join(folder, f)
                    return None

                paths = {
                    "name": name,
                    "fname": fname,
                    "img_ori": os.path.join(proj_dir, fname),
                    "img_eq": find_file(os.path.join(self.cfg.DIR_EQ, name)),
                    "mask": find_file(os.path.join(self.cfg.DIR_PROJ_METAL, name)),
                    "gt": find_file(os.path.join(self.cfg.DIR_LABEL, name)),
                }
                if paths["gt"]:
                    samples.append(paths)
        return samples

    def _pad_image(self, img):
        padded = np.zeros((self.cfg.IMG_SIZE, self.cfg.IMG_SIZE), dtype=np.uint8)
        padded[:img.shape[0], :img.shape[1]] = img
        return padded

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        img_ori = cv2.imread(sample["img_ori"], cv2.IMREAD_GRAYSCALE)
        img_eq = cv2.imread(sample["img_eq"], cv2.IMREAD_GRAYSCALE) if sample["img_eq"] else img_ori
        mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE) if sample["mask"] else np.zeros_like(img_ori)
        gt = cv2.imread(sample["gt"], cv2.IMREAD_GRAYSCALE)

        if img_ori is None or mask is None or gt is None:
            return self.__getitem__((idx + 1) % len(self))

        # Save original size
        h, w = img_ori.shape[:2]

        # Resize
        img_ori = self.rsz.apply_image(img_ori)
        img_eq = self.rsz.apply_image(img_eq)
        resized_hw = img_ori.shape[:2]
        mask = resize_binary_mask(mask, resized_hw)
        gt = resize_binary_mask(gt, resized_hw)

        # Pad
        img_ori = self._pad_image(img_ori)
        img_eq = self._pad_image(img_eq)
        mask = self._pad_image(mask)
        gt = self._pad_image(gt)

        # To tensor
        img_ori_t = torch.from_numpy(img_ori / 255.0).float().unsqueeze(0).unsqueeze(0)
        img_eq_t = torch.from_numpy(img_eq / 255.0).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        gt_np = (gt > 0).astype(np.float32)

        return {
            "sample": sample,
            "img_ori": img_ori_t,
            "img_eq": img_eq_t,
            "mask": mask_t,
            "gt": gt_np,
            "orig_hw": (h, w),
            "resized_hw": resized_hw,
        }
