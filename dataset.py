import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms



FERPLUS_EMOTION_COLUMNS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
FERPLUS_COLUMN_TO_CLASS = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}
EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EXPRESSION_NAMES = ["Shocked", "Happy", "Neutral"]
REQUIRED_COLUMNS = {"emotion", "pixels", "Usage"}
VALID_SPLITS = {"Training", "PublicTest", "PrivateTest"}
VALID_LABEL_MODES = {"emotion", "expression"}

EXPRESSION_MAP = {
    0: "Shocked",
    1: "Shocked",
    2: "Shocked",
    3: "Happy",
    4: "Shocked",
    5: "Shocked",
    6: None,
}

EMOTION_TO_EXPRESSION_CLASS = {
    0: 0,  # Angry -> Shocked
    1: 0,  # Disgust -> Shocked
    2: 0,  # Fear -> Shocked
    3: 1,  # Happy -> Happy
    4: 0,  # Sad -> Shocked
    5: 0,  # Surprise -> Shocked
    6: 2,  # Neutral -> Neutral/no monkey
}

EXPRESSION_CLASS_TO_MONKEY = {
    0: "Shocked",
    1: "Happy",
    2: None,
}


def load_fer2013_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"{csv_path} is not a valid FER2013 CSV. Missing columns: {missing_cols}. "
            "Download the Kaggle msambare/fer2013 dataset and extract fer2013.csv."
        )
    return df


def load_ferplus_labels(ferplus_csv_path):
    """Load FERPlus labels from fer2013new.csv.
    Returns numpy array of majority-vote emotion labels (0-6) same length as FER2013 rows."""
    df = pd.read_csv(ferplus_csv_path)
    df.columns = df.columns.str.strip()
    labels = []
    for _, row in df.iterrows():
        votes = {c: int(row[c]) for c in FERPLUS_EMOTION_COLUMNS}
        max_emo = max(FERPLUS_EMOTION_COLUMNS, key=lambda c: votes[c])
        if votes[max_emo] == 0:
            labels.append(6)  # neutral fallback
        else:
            labels.append(FERPLUS_COLUMN_TO_CLASS[max_emo])
    return np.array(labels, dtype=np.int64)

def get_class_names(label_mode="expression"):
    if label_mode == "emotion":
        return EMOTION_NAMES
    if label_mode == "expression":
        return EXPRESSION_NAMES
    raise ValueError(f"Unknown label_mode '{label_mode}'. Expected one of: {sorted(VALID_LABEL_MODES)}")


def get_monkey_map(label_mode="expression"):
    if label_mode == "emotion":
        return EXPRESSION_MAP
    if label_mode == "expression":
        return EXPRESSION_CLASS_TO_MONKEY
    raise ValueError(f"Unknown label_mode '{label_mode}'. Expected one of: {sorted(VALID_LABEL_MODES)}")


def convert_labels(emotion_ids, label_mode="expression"):
    emotion_ids = np.asarray(emotion_ids, dtype=np.int64)
    if label_mode == "emotion":
        return emotion_ids
    if label_mode == "expression":
        mapper = np.vectorize(EMOTION_TO_EXPRESSION_CLASS.__getitem__)
        return mapper(emotion_ids).astype(np.int64)
    raise ValueError(f"Unknown label_mode '{label_mode}'. Expected one of: {sorted(VALID_LABEL_MODES)}")


def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split="Training", transform=None, label_mode="expression",
                 cache_pixels=True, ferplus_csv=None):
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Expected one of: {sorted(VALID_SPLITS)}")
        if label_mode not in VALID_LABEL_MODES:
            raise ValueError(f"Unknown label_mode '{label_mode}'. Expected one of: {sorted(VALID_LABEL_MODES)}")

        full_df = load_fer2013_csv(csv_path)
        mask = full_df["Usage"] == split
        df = full_df[mask].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No FER2013 rows found for split '{split}' in {csv_path}")

        if ferplus_csv is not None:
            ferplus_labels = load_ferplus_labels(ferplus_csv)
            self.labels = convert_labels(ferplus_labels[mask.values], label_mode)
        else:
            self.labels = convert_labels(df["emotion"].values, label_mode)
        self.pixels = df["pixels"].values
        self.transform = transform
        self.label_mode = label_mode
        self.cache_pixels = cache_pixels

        if cache_pixels:
            parsed = [np.fromstring(row, dtype=np.uint8, sep=" ") for row in self.pixels]
            bad_rows = [i for i, values in enumerate(parsed) if values.size != 48 * 48]
            if bad_rows:
                first_bad = bad_rows[0]
                raise ValueError(
                    f"FER2013 row {first_bad} in split '{split}' has "
                    f"{parsed[first_bad].size} pixels, expected 2304"
                )
            self.images = np.stack(parsed).reshape(-1, 48, 48)
        else:
            self.images = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.images is None:
            pixel_values = np.fromstring(self.pixels[idx], dtype=np.uint8, sep=" ")
            if pixel_values.size != 48 * 48:
                raise ValueError(f"FER2013 row {idx} has {pixel_values.size} pixels, expected 2304")
            image = pixel_values.reshape(48, 48)
        else:
            image = self.images[idx]
        image = Image.fromarray(image, mode="L").convert("RGB")

        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_class_weights(csv_path, split="Training", label_mode="expression", ferplus_csv=None):
    full_df = load_fer2013_csv(csv_path)
    mask = full_df["Usage"] == split
    df = full_df[mask]
    if df.empty:
        raise ValueError(f"No FER2013 rows found for split '{split}' in {csv_path}")

    if ferplus_csv is not None:
        ferplus_labels = load_ferplus_labels(ferplus_csv)
        labels = convert_labels(ferplus_labels[mask.values], label_mode)
    else:
        labels = convert_labels(df["emotion"].values, label_mode)
    num_classes = len(get_class_names(label_mode))
    counts = pd.Series(labels).value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    total = counts.sum()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present = counts > 0
    present_weights = (total / (num_classes * counts[present])).astype(float).tolist()
    weights[present.tolist()] = torch.tensor(present_weights, dtype=torch.float32)
    return weights
