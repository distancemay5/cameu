import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cameu import CUBDataset, CaMeU


def main():
    parser = argparse.ArgumentParser(description="Evaluate CaMeU on CUB")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--recognizer-config", default="configs/resnet50_cub.py")
    parser.add_argument("--recognizer-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/cameu/result.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU")
    device = torch.device("cuda:0")
    dataset = CUBDataset(args.data_root, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = CaMeU(args.recognizer_config, args.recognizer_checkpoint).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = [key for key in missing if key.startswith("lpn.backbone.")]
    if len(allowed) != len(missing) or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    correct = 0
    total = 0
    psnr_sum = 0.0
    with torch.inference_mode():
        for _, clear, restored, labels, _ in tqdm(loader, desc="evaluate"):
            clear = clear.to(device, non_blocking=True)
            restored = restored.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(restored, stages=2)
            logits = model.recognizer(output, return_cls=True)["logits"]
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
            mse = (output - clear).square().flatten(1).mean(1).clamp_min(1e-12)
            psnr_sum += (-10.0 * torch.log10(mse)).sum().item()

    result = {
        "samples": total,
        "correct": correct,
        "top1": 100.0 * correct / total,
        "psnr": psnr_sum / total,
        "checkpoint": str(args.checkpoint),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
