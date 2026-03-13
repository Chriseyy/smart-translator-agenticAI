# watch -n 0.5 nvidia-smi
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RVLCDIPDataset(Dataset):
    """PyTorch dataset for RVL-CDIP images indexed by (path, label) pairs.

    Args:
        index: List of (filepath, label) tuples.
        image_size: Target crop size (square).
        train: Whether to use train-time transforms.
    """

    def __init__(self, index, image_size, train=False):
        self.index = index

        # Aktuell identisch für train/eval (kann später leicht erweitert werden).
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])

    def __len__(self):
        """Returns the dataset size."""
        return len(self.index)

    def __getitem__(self, idx):
        """Loads and returns one sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple (image_tensor, label_tensor).
        """
        path, label = self.index[idx]
        # PIL-Open im Context-Manager, damit Filehandles sauber schließen.
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.tf(img)
        return img, torch.tensor(label, dtype=torch.long)


def data_loader(n_train=1000, n_val=200, n_test=200, image_size=224, batch_size=32):
    """Creates train/val/test DataLoaders for a RVL-CDIP subset.

    Args:
        n_train: Number of training samples to load from index.
        n_val: Number of validation samples to load from index.
        n_test: Number of test samples to load from index.
        image_size: Center crop size used in transforms.
        batch_size: Batch size for training loader (val/test use 2x).

    Returns:
        Tuple (train_dl, val_dl, test_dl).
    """
    root_dir = "data/rvl-cdip"
    labels_dir = os.path.join(root_dir, "labels")

    def read_index(label_file, n):
        """Reads up to n lines from an index file into (abs_path, label) items."""
        items = []
        with open(label_file, "r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                path, label = line.strip().split()
                items.append((os.path.join(root_dir, "images", path), int(label)))
        return items

    # Splits über die provided txt-Files ziehen.
    train_index = read_index(os.path.join(labels_dir, "train.txt"), n_train)
    val_index   = read_index(os.path.join(labels_dir, "val.txt"),   n_val)
    test_index  = read_index(os.path.join(labels_dir, "test.txt"),  n_test)

    # Datasets bauen.
    train_ds = RVLCDIPDataset(train_index, image_size, train=True)
    val_ds   = RVLCDIPDataset(val_index,   image_size, train=False)
    test_ds  = RVLCDIPDataset(test_index,  image_size, train=False)

    # Loader-Performance-Settings.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nw = 8
    pin = (device == "cuda")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,        # no multiprocessing
        pin_memory=False,     # no pinned memory
    )

    return train_dl, val_dl, test_dl
