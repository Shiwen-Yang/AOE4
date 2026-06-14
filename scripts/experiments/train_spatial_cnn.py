"""Train a CNN on building placement heatmaps and extract spatial embeddings.

Usage:
    # Sanity check: visualize N heatmap pairs as PNG
    python scripts/experiments/train_spatial_cnn.py --dry-run --visualize 5 \\
        --snapshots data/realtime_outcome_prediction/features/v3_all/snapshots.parquet \\
        --parsed-dir data/replays/parsed

    # Full training run
    python scripts/experiments/train_spatial_cnn.py \\
        --snapshots data/realtime_outcome_prediction/features/v3_all/snapshots.parquet \\
        --parsed-dir data/replays/parsed \\
        --output-dir data/realtime_outcome_prediction/features/v4_spatial
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from realtime_outcome_prediction.config import (
    CACHE_DIR,
    FEATURE_DIR,
    PARSED_REPLAY_DIR,
)
from realtime_outcome_prediction.metadata import build_pbgid_index, load_or_update_aoe4world_repo
from realtime_outcome_prediction.spatial import (
    GRID_SIZE,
    N_CHANNELS,
    precompute_heatmaps,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset
    _F = F  # alias for use in main()

    _TORCH_OK = True
except ImportError as _e:
    _TORCH_OK = False
    _IMPORT_ERROR = str(_e)


class PlayerEncoder(nn.Module):
    """Shared CNN encoder: (8, 64, 64) → 64-dim embedding."""

    def __init__(self, in_channels: int = N_CHANNELS, embed_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),   # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),            # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # (64, 32, 32)
            nn.Conv2d(64, 64, 3, padding=1),            # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # (64, 16, 16)
            nn.AdaptiveAvgPool2d(1),                    # (64, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),                               # (64,)
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class PairHead(nn.Module):
    """Antisymmetric pair head: produces P(slot1 wins | both embeddings).

    logit = score(emb1, emb2) − score(emb2, emb1)   [same linear weights]
    → sigmoid(logit) = P(slot1 wins), and P(slot1,slot2) + P(slot2,slot1) = 1 exactly.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        score_12 = self.net(torch.cat([emb1, emb2], dim=-1)).squeeze(-1)
        score_21 = self.net(torch.cat([emb2, emb1], dim=-1)).squeeze(-1)
        return score_12 - score_21  # raw logit (no sigmoid)


class BuildingCNN(nn.Module):
    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.encoder = PlayerEncoder(N_CHANNELS, embed_dim)
        self.pair_head = PairHead(embed_dim)

    def forward(
        self, h1: torch.Tensor, h2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb1 = self.encoder(h1)
        emb2 = self.encoder(h2)
        logit = self.pair_head(emb1, emb2)
        return logit, emb1, emb2


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _auc(logits: np.ndarray, targets: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return float("nan")
    probs = 1.0 / (1.0 + np.exp(-logits))
    return float(roc_auc_score(targets, probs))


def _run_epoch(
    model: BuildingCNN,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    train: bool,
    augment: bool = False,
) -> tuple[float, float]:
    """Return (mean BCE loss, AUC)."""
    model.train(train)
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_targets = [], []
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for h1, h2, target, _ in loader:
            h1 = h1.to(device)
            h2 = h2.to(device)
            target = target.to(device)

            # 50% random swap augmentation (map mirror symmetry)
            if augment and train:
                mask = torch.rand(h1.size(0), device=device) < 0.5
                h1[mask], h2[mask] = h2[mask].clone(), h1[mask].clone()
                target[mask] = 1.0 - target[mask]

            logit, _, _ = model(h1, h2)
            loss = criterion(logit, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * h1.size(0)
            all_logits.append(logit.detach().cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    n = len(all_targets)
    return total_loss / n, _auc(all_logits, all_targets)


@torch.no_grad()
def _extract_embeddings(
    model: BuildingCNN,
    loader: DataLoader,
    device: torch.device,
    records: list[dict],
) -> list[dict]:
    model.eval()
    rows = []
    idx = 0
    for h1, h2, target, minute in loader:
        h1, h2 = h1.to(device), h2.to(device)
        _, emb1, emb2 = model(h1, h2)
        emb1 = emb1.cpu().numpy()
        emb2 = emb2.cpu().numpy()
        bs = h1.size(0)
        for i in range(bs):
            r = records[idx + i]
            row = {
                "replay_id": r["replay_id"],
                "snapshot_minute": r["snapshot_minute"],
                "split": r.get("split"),
                "target": float(target[i].item()),
            }
            for j in range(emb1.shape[1]):
                row[f"spatial_emb1_{j}"] = float(emb1[i, j])
                row[f"spatial_emb2_{j}"] = float(emb2[i, j])
            rows.append(row)
        idx += bs
    return rows


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _visualize(records: list[dict], cache_dir: Path, n: int, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping visualisation")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    channel_names = ["Eco", "Mil prod", "Defensive", "Landmarks", "Valid mask"]
    sample = records[:n]
    for rec in sample:
        rid, minute = rec["replay_id"], rec["snapshot_minute"]
        path = cache_dir / f"{rid}_{minute}.npy"
        if not path.exists():
            print(f"  cache missing: {path.name}")
            continue
        combined = np.load(str(path))  # (2, 5, G, G)
        fig, axes = plt.subplots(2, N_CHANNELS, figsize=(N_CHANNELS * 2.5, 5))
        for slot_idx, slot_label in enumerate(("Slot 1", "Slot 2")):
            for ch in range(N_CHANNELS):
                ax = axes[slot_idx, ch]
                ax.imshow(combined[slot_idx, ch], origin="upper", cmap="hot")
                ax.set_title(channel_names[ch], fontsize=7)
                ax.axis("off")
            axes[slot_idx, 0].set_ylabel(slot_label, fontsize=8)
        fig.suptitle(f"replay {rid}  minute {minute}", fontsize=9)
        fig.tight_layout()
        save_path = out_dir / f"heatmap_{rid}_{minute}.png"
        fig.savefig(str(save_path), dpi=100)
        plt.close(fig)
        print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Symmetry check
# ---------------------------------------------------------------------------


@torch.no_grad()
def _check_symmetry(model: BuildingCNN, loader: DataLoader, device: torch.device) -> float:
    """Return mean |P(A,B) + P(B,A) - 1|; should be < 0.001 by construction."""
    model.eval()
    errors = []
    for h1, h2, _, _ in loader:
        h1, h2 = h1.to(device), h2.to(device)
        logit_ab, _, _ = model(h1, h2)
        logit_ba, _, _ = model(h2, h1)
        p_ab = torch.sigmoid(logit_ab)
        p_ba = torch.sigmoid(logit_ba)
        errors.append((p_ab + p_ba - 1.0).abs().cpu().numpy())
    return float(np.concatenate(errors).mean())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN on building placement heatmaps")
    p.add_argument(
        "--snapshots",
        default=str(FEATURE_DIR.parent / "v3_all" / "snapshots.parquet"),
    )
    p.add_argument("--parsed-dir", default=str(PARSED_REPLAY_DIR))
    p.add_argument("--output-dir", default=str(FEATURE_DIR.parent / "v4_spatial"))
    p.add_argument("--cache-dir", default=str(CACHE_DIR / "spatial_heatmaps"))
    p.add_argument("--aoe4world-repo-dir", default=None)
    p.add_argument("--grid-size", type=int, default=GRID_SIZE)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--force-recompute", action="store_true", help="Recompute heatmaps even if cached")
    p.add_argument("--dry-run", action="store_true", help="Skip training; only precompute and optionally visualise")
    p.add_argument("--visualize", type=int, default=0, metavar="N", help="Render N heatmap pairs as PNG")
    p.add_argument("--no-augment", action="store_true", help="Disable swap augmentation")
    return p.parse_args(argv)


def main(argv=None) -> None:
    if not _TORCH_OK:
        sys.exit(f"PyTorch / sklearn not available: {_IMPORT_ERROR}")

    args = _parse_args(argv)
    snapshots_path = Path(args.snapshots)
    parsed_dir = Path(args.parsed_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Build pbgid index
    print("Loading AoE4 World metadata...")
    aoe4world_dir = Path(args.aoe4world_repo_dir) if args.aoe4world_repo_dir else None
    metadata = load_or_update_aoe4world_repo(CACHE_DIR, repo_dir=aoe4world_dir, update=False)
    pbgid_index = build_pbgid_index(metadata)
    print(f"  pbgid_index: {len(pbgid_index)} entries")

    # Precompute heatmaps
    print(f"Precomputing heatmaps → {cache_dir}")
    t0 = time.time()
    counts = precompute_heatmaps(
        snapshots_path=snapshots_path,
        parsed_dir=parsed_dir,
        pbgid_index=pbgid_index,
        cache_dir=cache_dir,
        grid_size=args.grid_size,
        force=args.force_recompute,
    )
    print(f"  {counts}  ({time.time() - t0:.1f}s)")

    # Optionally visualise
    if args.visualize > 0:
        print(f"Visualising {args.visualize} samples...")
        import pandas as pd
        _idx = pd.read_parquet(str(cache_dir / "heatmap_index.parquet"))
        _vis_records = _idx.to_dict("records")
        _visualize(_vis_records, cache_dir, args.visualize, output_dir / "heatmap_vis")

    if args.dry_run:
        print("--dry-run: exiting before training")
        return

    # Build consolidated array from individual cache files if needed
    consolidated = cache_dir / "all_heatmaps.npy"
    index_path = cache_dir / "heatmap_index.parquet"
    if not consolidated.exists() or not index_path.exists():
        import pandas as pd
        print("Building consolidated heatmap array...")
        t0 = time.time()
        snap_df = pd.read_parquet(snapshots_path, columns=["replay_id", "snapshot_minute", "target", "split"])
        snap_df = snap_df.drop_duplicates(subset=["replay_id", "snapshot_minute"]).reset_index(drop=True)
        arrays = []
        valid_rows = []
        for i, row in snap_df.iterrows():
            p = cache_dir / f"{row['replay_id']}_{row['snapshot_minute']}.npy"
            if p.exists():
                arrays.append(np.load(str(p)))
                valid_rows.append({**row.to_dict(), "array_idx": len(arrays) - 1})
        stacked = np.stack(arrays, axis=0)
        np.save(str(consolidated), stacked)
        idx_df = pd.DataFrame(valid_rows)
        idx_df.to_parquet(str(index_path), index=False)
        print(f"  shape={stacked.shape}  {stacked.nbytes/1e9:.2f} GB  ({time.time()-t0:.1f}s)")
        del arrays, stacked
    else:
        import pandas as pd

    print(f"Loading consolidated heatmap array from {consolidated}  ...")
    t0 = time.time()
    raw = np.load(str(consolidated))   # (N, 2, 8, 64, 64) at native 64×64
    idx_df = pd.read_parquet(str(index_path))
    print(f"  shape={raw.shape}  {raw.nbytes/1e9:.2f} GB  ({time.time()-t0:.1f}s)")

    # Optionally downsample to a smaller grid for faster training
    native_grid = raw.shape[-1]  # 64
    train_grid = args.grid_size   # e.g. 32
    if train_grid != native_grid:
        print(f"  Downsampling {native_grid}×{native_grid} → {train_grid}×{train_grid} ...")
        n = raw.shape[0]
        raw_t = torch.from_numpy(raw).view(n * 2 * N_CHANNELS, 1, native_grid, native_grid).float()
        raw_t = _F.avg_pool2d(raw_t, kernel_size=native_grid // train_grid)
        raw = raw_t.view(n, 2, N_CHANNELS, train_grid, train_grid).numpy().astype(np.float32)
        print(f"  Downsampled shape={raw.shape}  {raw.nbytes/1e9:.2f} GB")

    def _make_split_tensors(split_name):
        mask = idx_df["split"] == split_name
        rows = idx_df[mask].reset_index(drop=True)
        arr_idx = rows["array_idx"].values
        data = torch.from_numpy(raw[arr_idx])     # (N, 2, 8, 64, 64)
        h1 = data[:, 0]                           # (N, 8, 64, 64)
        h2 = data[:, 1]
        targets = torch.tensor(rows["target"].values, dtype=torch.float32)
        minutes = torch.tensor(rows["snapshot_minute"].values, dtype=torch.int32)
        records = rows.to_dict("records")
        return h1, h2, targets, minutes, records

    print("Building split tensors...")
    tr_h1, tr_h2, tr_t, tr_m, train_records = _make_split_tensors("train")
    va_h1, va_h2, va_t, va_m, valid_records = _make_split_tensors("valid")
    te_h1, te_h2, te_t, te_m, test_records = _make_split_tensors("test")
    print(f"  train={len(train_records)}  valid={len(valid_records)}  test={len(test_records)}")

    def _make_loader(h1, h2, targets, minutes, shuffle):
        ds = TensorDataset(h1, h2, targets, minutes)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(tr_h1, tr_h2, tr_t, tr_m, shuffle=True)
    valid_loader = _make_loader(va_h1, va_h2, va_t, va_m, shuffle=False)
    test_loader  = _make_loader(te_h1, te_h2, te_t, te_m, shuffle=False)

    all_records = train_records + valid_records + test_records
    del raw  # free ~6.6 GB once tensors are built

    # Model + optimiser
    model = BuildingCNN(embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_valid_auc = -1.0
    patience_left = args.patience
    best_state = None
    history = []

    print(f"Training for up to {args.max_epochs} epochs (patience={args.patience})...")
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = _run_epoch(
            model, train_loader, optimizer, device, train=True, augment=not args.no_augment
        )
        va_loss, va_auc = _run_epoch(model, valid_loader, optimizer, device, train=False)
        elapsed = time.time() - t0
        print(
            f"  epoch {epoch:3d}  "
            f"train loss={tr_loss:.4f} auc={tr_auc:.4f}  "
            f"valid loss={va_loss:.4f} auc={va_auc:.4f}  "
            f"({elapsed:.1f}s)"
        )
        row = {"epoch": epoch, "train_loss": tr_loss, "train_auc": tr_auc,
               "valid_loss": va_loss, "valid_auc": va_auc}
        history.append(row)

        if va_auc > best_valid_auc:
            best_valid_auc = va_auc
            patience_left = args.patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch} (best valid AUC={best_valid_auc:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    _, te_auc = _run_epoch(model, test_loader, optimizer, device, train=False)
    print(f"\nFinal: valid AUC={best_valid_auc:.4f}  test AUC={te_auc:.4f}")

    # Symmetry check
    sym_err = _check_symmetry(model, valid_loader, device)
    print(f"Symmetry check (valid): mean |P(A,B)+P(B,A)-1| = {sym_err:.6f}")
    if sym_err > 0.001:
        print("  WARNING: symmetry error > 0.001")

    # Save model
    model_path = output_dir / "spatial_cnn.pt"
    torch.save({"model_state": best_state, "embed_dim": args.embed_dim,
                "grid_size": args.grid_size, "n_channels": N_CHANNELS}, str(model_path))
    print(f"Saved model → {model_path}")

    # Extract embeddings for all splits
    print("Extracting embeddings...")
    _train_det = _make_loader(tr_h1, tr_h2, tr_t, tr_m, shuffle=False)
    all_rows = (
        _extract_embeddings(model.to(device), _train_det, device, train_records)
        + _extract_embeddings(model.to(device), valid_loader, device, valid_records)
        + _extract_embeddings(model.to(device), test_loader, device, test_records)
    )

    try:
        import pandas as pd
        emb_df = pd.DataFrame(all_rows)
        emb_path = output_dir / "embeddings.parquet"
        emb_df.to_parquet(str(emb_path), index=False)
        print(f"Saved embeddings → {emb_path}  shape={emb_df.shape}")
    except ImportError:
        import csv
        emb_path = output_dir / "embeddings.csv"
        with open(emb_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved embeddings (csv) → {emb_path}")

    # Save training history + meta
    meta = {
        "best_valid_auc": best_valid_auc,
        "test_auc": te_auc,
        "symmetry_error": sym_err,
        "n_params": n_params,
        "embed_dim": args.embed_dim,
        "grid_size": args.grid_size,
        "n_channels": N_CHANNELS,
        "history": history,
    }
    (output_dir / "spatial_cnn_meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
