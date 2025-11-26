# ---------------------------------------------------------------
# src/train/gradcam.py
# Compute Grad-CAM for CombinedModel (image + metadata)
# ---------------------------------------------------------------

import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# Make src/ importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset.isic_dataset import ISIC2019Dataset
from train.lightning_module import ISICLightningModule


# ------------------------------------------------------------------
# 1. Load & clean checkpoint
# ------------------------------------------------------------------
def load_clean_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    remove_keys = [
        k for k in ckpt["state_dict"]
        if ("class_weights" in k) or ("criterion" in k)
    ]

    for k in remove_keys:
        print(f"Removing: {k}")
        ckpt["state_dict"].pop(k)

    return ckpt


# ------------------------------------------------------------------
# 2. Wrapper model for Grad-CAM
# ------------------------------------------------------------------
class GradCAMWrapper(torch.nn.Module):
    def __init__(self, lightning_module, metadata_tensor):
        super().__init__()
        # Extract ONLY the underlying CombinedModel
        self.model = lightning_module.model     # <--- IMPORTANT FIX
        self.metadata = metadata_tensor

    def forward(self, x):
        b = x.size(0)
        meta = self.metadata.repeat(b, 1).to(x.device)
        return self.model(x, meta)



# ------------------------------------------------------------------
# 3. Save comparison image
# ------------------------------------------------------------------
def save_side_by_side(image_np, cam_np, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(image_np)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(cam_np)
    ax[1].set_title("Grad-CAM")
    ax[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 4. Main routine
# ------------------------------------------------------------------
def run_gradcam(
    ckpt_path="checkpoints/best-model.ckpt",
    num_samples=10,
    output_dir="gradcam_outputs",
):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading checkpoint...")
    ckpt = load_clean_checkpoint(ckpt_path)

    print("Rebuilding model...")
    model = ISICLightningModule(**ckpt["hyper_parameters"])
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval().cuda()

    # ------------------------------------------------------------------
    # Load metadata encoders EXACTLY as in training
    # ------------------------------------------------------------------
    proc_root = ROOT / "processed"
    train_csv = proc_root / "splits/train.csv"
    val_csv = proc_root / "splits/val.csv"
    img_dir = proc_root / "images_224_nohair"

    train_df = torch.load(train_csv, weights_only=False) if False else None
    import pandas as pd
    train_df = pd.read_csv(train_csv)

    age_min = train_df["age_approx"].min()
    age_max = train_df["age_approx"].max()

    sex_categories = ["male", "female", "unknown"]

    site_categories = list(train_df["anatom_site_general"].unique())
    if "unknown" not in site_categories:
        site_categories.append("unknown")

    # SAME val transforms as training
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Build dataset correctly
    val_dataset = ISIC2019Dataset(
        csv_path=val_csv,
        images_dir=img_dir,
        transform=val_tf,
        age_min=age_min,
        age_max=age_max,
        sex_categories=sex_categories,
        site_categories=site_categories,
    )

    print(f"\nGenerating Grad-CAM for {num_samples} samples...\n")

    # ------------------------------------------------------------------
    # Do Grad-CAM for N samples
    # ------------------------------------------------------------------
    for idx in range(num_samples):

        image, meta, label = val_dataset[idx]

        img_tensor = image.unsqueeze(0).cuda()
        meta_tensor = meta.unsqueeze(0).cuda()

        # Convert tensor → numpy for visualization
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Wrap model
        wrapped = GradCAMWrapper(model, meta_tensor).cuda()

       # Choose the correct final convolution layer of ResNet50
        target_layers = [
            wrapped.model.model.backbone.layer4[-1].conv3
        ]

        cam = GradCAMPlusPlus(
            model=wrapped,
            target_layers=target_layers
        )

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=None,
            eigen_smooth=True,
        )[0]



        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        out_path = output_dir / f"gradcam_{idx}.png"
        save_side_by_side(img_np, cam_image, out_path)

        print(f"Saved → {out_path}")

    print("\nDone. Grad-CAM images saved to:", output_dir)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best-model.ckpt")
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    run_gradcam(
        ckpt_path=args.ckpt,
        num_samples=args.num,
    )
