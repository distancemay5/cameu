from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESIZE = Resize((256, 256))


def find_image(root, relative_path):
    direct = root / relative_path
    if direct.exists():
        return direct
    for suffix in sorted(EXTENSIONS):
        candidate = root / relative_path.parent / f"{relative_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No paired image for {relative_path} under {root}")


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


class CUBDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.degraded_root = root / "degraded" / split
        self.clear_root = root / "clear" / split
        self.restored_root = root / "restored" / split
        if not self.degraded_root.is_dir():
            self.degraded_root = root / "fog" / split
        if not self.restored_root.is_dir():
            self.restored_root = root / "dehaze" / split
        for path in (self.degraded_root, self.clear_root, self.restored_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        classes = sorted(path.name for path in self.degraded_root.iterdir() if path.is_dir())
        if len(classes) != 200:
            raise RuntimeError(f"Expected 200 classes, found {len(classes)}")
        class_to_idx = {name: index for index, name in enumerate(classes)}
        self.samples = []
        for class_name in classes:
            for degraded_path in (self.degraded_root / class_name).rglob("*"):
                if degraded_path.is_file() and degraded_path.suffix.lower() in EXTENSIONS:
                    relative = degraded_path.relative_to(self.degraded_root)
                    self.samples.append((
                        degraded_path,
                        find_image(self.clear_root, relative),
                        find_image(self.restored_root, relative),
                        class_to_idx[class_name],
                        str(relative),
                    ))
        if not self.samples:
            raise RuntimeError(f"No images found under {self.degraded_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        degraded_path, clear_path, restored_path, label, relative = self.samples[index]
        degraded = read_rgb(degraded_path)
        clear = read_rgb(clear_path)
        restored = read_rgb(restored_path)
        if clear.shape[-2:] != degraded.shape[-2:]:
            clear = torch.from_numpy(cv2.resize(
                clear.permute(1, 2, 0).numpy(), degraded.shape[-2:][::-1], interpolation=cv2.INTER_LINEAR
            )).permute(2, 0, 1).contiguous()
        if restored.shape[-2:] != degraded.shape[-2:]:
            restored = torch.from_numpy(cv2.resize(
                restored.permute(1, 2, 0).numpy(), degraded.shape[-2:][::-1], interpolation=cv2.INTER_LINEAR
            )).permute(2, 0, 1).contiguous()
        return RESIZE(degraded), RESIZE(clear), RESIZE(restored), label, relative
