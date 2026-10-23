import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from cameu import CUBDataset, CaMeU, CaMeULoss


def distributed_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), world_size


def trainable_state_dict(model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("lpn.backbone.")
    }


def main():
    parser = argparse.ArgumentParser(description="Train CaMeU on CUB")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--recognizer-config", default="configs/resnet50_cub.py")
    parser.add_argument("--recognizer-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/cameu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    local_rank, rank, world_size = distributed_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CaMeU training requires a CUDA GPU")
    device = torch.device("cuda", local_rank)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = CUBDataset(args.data_root, "train")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = CaMeU(args.recognizer_config, args.recognizer_checkpoint).to(device)
    criterion = CaMeULoss(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        metadata = vars(args) | {
            "world_size": world_size,
            "effective_batch_size": args.batch_size * world_size,
            "train_samples": len(dataset),
            "trainable_parameters": sum(p.numel() for p in trainable),
        }
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        iterator = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}") if rank == 0 else loader
        running = 0.0
        for _, clear, restored, labels, _ in iterator:
            clear = clear.to(device, non_blocking=True)
            restored = restored.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            prediction, states, final_logits = model(
                restored, return_training_state=True
            )
            loss, metrics = criterion(prediction, states, final_logits, clear, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            running += loss.detach().item()
            if rank == 0:
                iterator.set_postfix(
                    loss=f"{metrics['total'].item():.4f}",
                    task=f"{metrics['recognition'].item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
        scheduler.step()

        mean_loss = torch.tensor(running / len(loader), device=device)
        if world_size > 1:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size
        if rank == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model": trainable_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "mean_loss": mean_loss.item(),
            }
            torch.save(checkpoint, output / "latest.pth")
            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint, output / f"epoch_{epoch + 1}.pth")
            print(f"epoch={epoch + 1} mean_loss={mean_loss.item():.6f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
