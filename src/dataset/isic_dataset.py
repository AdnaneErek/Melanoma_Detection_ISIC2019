import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


# ============================================================
# Helper functions for metadata encoding
# ============================================================

def compute_age_norm_params(train_df):
    """Compute min/max age from TRAIN ONLY."""
    age_min = train_df["age_approx"].min()
    age_max = train_df["age_approx"].max()
    return float(age_min), float(age_max)


def normalize_age(age, age_min, age_max):
    """Normalize age to [0, 1]."""
    if np.isnan(age):
        return 0.0
    return (age - age_min) / (age_max - age_min + 1e-8)


def build_sex_categories(df):
    """Return ordered categories: male, female, unknown."""
    cats = ["male", "female"]
    return cats + ["unknown"]


def encode_sex(sex, categories):
    sex = str(sex).lower()
    if sex not in categories:
        sex = "unknown"
    vec = np.zeros(len(categories), dtype=np.float32)
    vec[categories.index(sex)] = 1.0
    return vec


def build_site_categories(df):
    """Make one-hot categories for anatom_site_general."""
    sites = list(df["anatom_site_general"].unique())
    if "unknown" not in sites:
        sites.append("unknown")
    return sites


def encode_site(site, site_categories):
    site = str(site)
    if site not in site_categories:
        site = "unknown"
    vec = np.zeros(len(site_categories), dtype=np.float32)
    idx = site_categories.index(site)
    vec[idx] = 1.0
    return vec


# ============================================================
# PyTorch Dataset
# ============================================================

class ISIC2019Dataset(torch.utils.data.Dataset):
    """
    Dataset returning:
        image_tensor (3,224,224)
        metadata_tensor (d_meta)
        label_tensor (int)
    """
    def __init__(
        self,
        csv_path,
        images_dir,
        transform=None,
        age_min=None,
        age_max=None,
        sex_categories=None,
        site_categories=None,
    ):
        """
        csv_path: train.csv / val.csv / test.csv
        images_dir: processed images folder (PNG)
        transform: torchvision transforms for images
        """
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Store metadata encoders
        self.age_min = age_min
        self.age_max = age_max
        self.sex_categories = sex_categories
        self.site_categories = site_categories

    def __len__(self):
        return len(self.df)

    def encode_metadata(self, row):
        # Age normalized to [0,1]
        age = normalize_age(row["age_approx"], self.age_min, self.age_max)
        age_feat = np.array([age], dtype=np.float32)

        # Sex → one-hot
        sex_feat = encode_sex(row["sex"], self.sex_categories)

        # Anatomical site → one-hot
        site_feat = encode_site(row["anatom_site_general"], self.site_categories)

        # Final vector
        return np.concatenate([age_feat, sex_feat, site_feat]).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -----------------------------------------------------
        # Load Image (already preprocessed to 224×224)
        # -----------------------------------------------------
        img_name = row["image_name"] + ".png"
        img_path = self.images_dir / img_name

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # -----------------------------------------------------
        # Metadata Vector
        # -----------------------------------------------------
        meta_vec = self.encode_metadata(row)
        meta_tensor = torch.tensor(meta_vec, dtype=torch.float32)

        # -----------------------------------------------------
        # Label
        # -----------------------------------------------------
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return img, meta_tensor, label
