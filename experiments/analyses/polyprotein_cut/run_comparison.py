#!/usr/bin/env python3
"""Leave-one-group-out ESM3Di vs stock ESM++ cut-site comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "Amarillovirales"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/esm3di_from_mlm/epoch_5.pt"
MODEL_NAME = "Synthyra/ESMplusplus_small"
MODEL_REVISION = "46c5f7d"
DEFAULT_SEEDS = (17,)
WINDOW_OVERLAP = 128
TOP_K = 10
TOP_K_TOLERANCE = 5


def parse_cut_sites(path: Path) -> dict[str, list[int]]:
    """Keep the notebook's ICTV rule: first residue of FT SITE /note=\"Cleavage\"."""
    by_id: dict[str, set[int]] = {}
    current_id = None
    pending_start = None
    pending_cleavage = False

    def commit():
        nonlocal pending_start, pending_cleavage
        if current_id and pending_start is not None and pending_cleavage:
            by_id.setdefault(current_id, set()).add(pending_start)
        pending_start = None
        pending_cleavage = False

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("ID"):
                commit()
                match = re.match(r"^ID\s+([^\s]+)", line)
                current_id = match.group(1) if match else None
            elif line.startswith("//"):
                commit()
                current_id = None
            elif current_id is not None:
                match = re.match(r"^FT\s+SITE\s+(\d+)(?:\.\.(\d+))?", line)
                if match:
                    commit()
                    pending_start = int(match.group(1))
                elif pending_start is not None and '/note="Cleavage"' in line:
                    pending_cleavage = True
                elif pending_start is not None and re.match(r"^FT\s{3}[A-Z]", line):
                    commit()
    commit()
    return {key: sorted(values) for key, values in by_id.items() if values}


def load_records(data_dir: Path = DATA_DIR) -> list[dict]:
    records = []
    seen = set()
    sites_by_id: dict[str, list[int]] = {}
    files = sorted(data_dir.glob("*/hpraise*_ICTV"))
    if len(files) != 4:
        raise RuntimeError(f"Expected 4 ICTV files under {data_dir}, found {len(files)}")
    for path in files:
        sites_by_id.update(parse_cut_sites(path))
        for block in path.read_text(encoding="utf-8", errors="replace").split("//"):
            if not block.strip():
                continue
            record = SeqIO.read(StringIO(block.strip() + "\n//"), "swiss")
            identifier = record.name
            if identifier in seen:
                raise ValueError(f"Duplicate ICTV sequence identifier: {identifier}")
            seen.add(identifier)
            sequence = str(record.seq).upper()
            # The full label map is applied after all source files have been parsed.
            records.append({
                "identifier": identifier,
                "accession": record.id,
                "group": path.parent.name,
                "sequence": sequence,
                "source_path": str(path),
            })
    for record in records:
        record["sites"] = sites_by_id.get(record["identifier"], [])
        if not record["sites"]:
            raise ValueError(f"No parsed cleavage labels for ICTV record {record['identifier']}")
        if record["sites"][-1] > len(record["sequence"]):
            raise ValueError(f"Cut site beyond sequence end: {record['identifier']}")
    if len(records) != 86:
        raise RuntimeError(f"Expected 86 annotated ICTV records, found {len(records)}")
    return records


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_extract_embeddings(records: list[dict], source: str, checkpoint: Path, cache_dir: Path,
                               device: str, batch_size: int, adapter: str | None = None) -> np.ndarray:
    from esm3di.inference.windowing import window_ranges, residue_window_size

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = (f"adapter:{adapter}" if adapter else digest_file(checkpoint)) if source == "esm3di" else None
    sequence_sha = hashlib.sha256("\n".join(r["sequence"] for r in records).encode()).hexdigest()
    metadata = {
        "source": source,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha,
        "sequence_sha256": sequence_sha,
        "overlap": WINDOW_OVERLAP,
    }
    name = f"embeddings_{source}.npy"
    data_path, meta_path = cache_dir / name, cache_dir / f"{name}.json"
    offsets = np.cumsum([0] + [len(r["sequence"]) for r in records])
    n_residues = int(offsets[-1])
    if data_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == metadata:
        matrix = np.load(data_path, mmap_mode="r")
        if matrix.shape == (n_residues, 960):
            print(f"Using cached {source} embeddings: {data_path}", flush=True)
            return matrix

    if source == "esm3di" and adapter is not None:
        from transformers import AutoModel
        from peft import PeftModel

        base = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)
        tokenizer = base.tokenizer
        model = PeftModel.from_pretrained(base, adapter).to(device).eval()

        def forward(input_ids, attention_mask):
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hs = getattr(out, "hidden_states", None)
            return hs[-1] if hs is not None else out.last_hidden_state

        max_len = 1024
    elif source == "esm3di":
        from esm3di._core.checkpoints import load_3di_checkpoint

        esm, ckpt = load_3di_checkpoint(str(checkpoint), device=device, hf_revision=MODEL_REVISION)
        model = esm.model
        tokenizer = esm.tokenizer

        def forward(input_ids, attention_mask):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            if output.hidden_states is None:
                raise RuntimeError("ESM3Di checkpoint did not return hidden states")
            return output.hidden_states[-1]

        max_len = int((ckpt.get("args") or {}).get("max_seq_length") or 1024)
    else:
        from transformers import AutoModel

        # Stock backbone at its pinned revision, with its own paired tokenizer.
        model = AutoModel.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True,
        )
        tokenizer = model.tokenizer
        model = model.to(device).eval()

        def forward(input_ids, attention_mask):
            output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            if output.hidden_states is None:
                raise RuntimeError("Stock ESM++ did not return hidden states")
            return output.hidden_states[-1]

        max_len = 1024

    window = min(max_len, residue_window_size(tokenizer) + tokenizer.num_special_tokens_to_add(pair=False))
    overlap = min(WINDOW_OVERLAP, (window - tokenizer.num_special_tokens_to_add(pair=False)) // 2)
    output = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32, shape=(n_residues, 960))
    started = time.time()
    for i, record in enumerate(records):
        sequence = record["sequence"]
        windows = window_ranges(len(sequence), window - tokenizer.num_special_tokens_to_add(pair=False), overlap)
        chunks = [sequence[left:right] for left, right in windows]
        sums = torch.zeros(len(sequence), 960)
        counts = torch.zeros(len(sequence), 1)
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            encoded = tokenizer(
                batch_chunks, return_tensors="pt", padding=True, truncation=False,
                add_special_tokens=True, return_special_tokens_mask=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            hidden = forward(input_ids, attention_mask).detach().float().cpu()
            residue_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
            for j, chunk in enumerate(batch_chunks):
                values = hidden[j][residue_mask[j]]
                if len(values) != len(chunk):
                    raise RuntimeError(f"Tokenizer residue mismatch for {record['identifier']}")
                left, right = windows[start + j]
                sums[left:right] += values
                counts[left:right] += 1
        if torch.any(counts == 0):
            raise RuntimeError(f"Windowing left uncovered residues in {record['identifier']}")
        output[offsets[i]:offsets[i + 1]] = (sums / counts).numpy()
        if i % 4 == 0 or i == len(records) - 1:
            output.flush()
            print(f"{source}: {i + 1}/{len(records)} sequences, {time.time() - started:.1f}s", flush=True)
    output.flush()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return np.load(data_path, mmap_mode="r")


class PositionDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32)).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class SmallCutSiteFFN(nn.Module):
    def __init__(self, d_in, dropout=0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class SmallCutSiteTransformer(nn.Module):
    def __init__(self, d_in, dropout=0.35):
        super().__init__()
        self.input_proj = nn.Linear(d_in, 256)
        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, mask):
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        return self.head(self.norm(h)).squeeze(-1)


def oscar_penalty(weights, lambda_l1=1e-6, lambda_pair=2e-7):
    magnitudes = weights.reshape(-1).abs()
    ordered = torch.sort(magnitudes, descending=True).values
    ranks = torch.arange(len(ordered) - 1, -1, -1, device=ordered.device, dtype=ordered.dtype)
    return lambda_l1 * magnitudes.sum() + lambda_pair * (ordered * ranks).sum()


def windows_for_classifier(records, features, offsets, mask_overlaps=True):
    # Each 512-residue example has 64 residues of overlap; every token is trained once
    # by masking overlap duplicates to keep the token loss unbiased.
    chunks = []
    for idx, record in enumerate(records):
        length = len(record["sequence"])
        start = 0
        while start < length:
            end = min(start + 512, length)
            x = np.asarray(features[offsets[idx] + start:offsets[idx] + end], dtype=np.float32)
            y = np.zeros(end - start, dtype=np.float32)
            for position in record["sites"]:
                if start < position <= end:
                    y[position - start - 1] = 1
            valid = np.ones(end - start, dtype=np.float32)
            if start and mask_overlaps:
                valid[:64] = 0
            chunks.append({"identifier": record["identifier"], "X": x, "y": y, "valid": valid, "group": record["group"], "start": start})
            if end == length:
                break
            start = end - 64
    return chunks


def batch_chunks(chunks):
    max_len = max(len(r["y"]) for r in chunks)
    d = chunks[0]["X"].shape[1]
    x = np.zeros((len(chunks), max_len, d), dtype=np.float32)
    y = np.zeros((len(chunks), max_len), dtype=np.float32)
    loss_mask = np.zeros((len(chunks), max_len), dtype=np.float32)
    attention_mask = np.zeros((len(chunks), max_len), dtype=np.float32)
    for i, row in enumerate(chunks):
        n = len(row["y"])
        x[i, :n] = row["X"]
        y[i, :n] = row["y"]
        loss_mask[i, :n] = row["valid"]
        attention_mask[i, :n] = 1
    return (torch.from_numpy(x), torch.from_numpy(y),
            torch.from_numpy(loss_mask), torch.from_numpy(attention_mask))


def train_predict(records, features, offsets, test_ids, architecture, seed, device, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = [i for i, r in enumerate(records) if r["group"] not in test_ids]
    test = [i for i, r in enumerate(records) if r["group"] in test_ids]
    train_sites = sum(len(records[i]["sites"]) for i in train)
    train_tokens = sum(len(records[i]["sequence"]) for i in train)
    pos_weight = (train_tokens - train_sites) / max(train_sites, 1)
    pos_weight_t = torch.tensor(pos_weight, device=device)

    if architecture == "ffn":
        xs = np.concatenate([features[offsets[i]:offsets[i + 1]] for i in train]).astype(np.float32)
        ys = np.concatenate([np.isin(np.arange(len(records[i]["sequence"])), np.asarray(records[i]["sites"]) - 1).astype(np.float32) for i in train])
        loader = DataLoader(PositionDataset(xs, ys), batch_size=2048, shuffle=True)
        model = SmallCutSiteFFN(960).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = nn.functional.dropout(xb, p=0.10, training=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss = loss + oscar_penalty(model.net[-1].weight)
                loss.backward()
                optimizer.step()
        model.eval()
        predictions = {}
        with torch.no_grad():
            for i in test:
                x = torch.from_numpy(np.asarray(features[offsets[i]:offsets[i + 1]], dtype=np.float32)).to(device)
                predictions[i] = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
        return predictions

    train_chunks = windows_for_classifier([records[i] for i in train],
        np.concatenate([features[offsets[i]:offsets[i + 1]] for i in train]),
        np.cumsum([0] + [len(records[i]["sequence"]) for i in train]))
    test_chunks_by_id = {}
    for i in test:
        local = windows_for_classifier([records[i]], features, np.asarray([offsets[i]]), mask_overlaps=False)
        test_chunks_by_id[i] = local
    model = SmallCutSiteTransformer(960).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_chunks))
        for start in range(0, len(order), 4):
            batch = [train_chunks[j] for j in order[start:start + 4]]
            xb, yb, loss_mask, attention_mask = batch_chunks(batch)
            xb, yb = xb.to(device), yb.to(device)
            loss_mask, attention_mask = loss_mask.to(device), attention_mask.to(device)
            xb = nn.functional.dropout(xb, p=0.08, training=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, attention_mask)
            per_token = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none", pos_weight=pos_weight_t)
            loss = (per_token * loss_mask).sum() / loss_mask.sum().clamp_min(1)
            loss = loss + oscar_penalty(model.head[-1].weight)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = {}
    with torch.no_grad():
        for i, chunks in test_chunks_by_id.items():
            probs = np.zeros(len(records[i]["sequence"]), dtype=np.float32)
            counts = np.zeros_like(probs)
            for start in range(0, len(chunks), 4):
                batch = chunks[start:start + 4]
                xb, _, _, attention_mask = batch_chunks(batch)
                scores = torch.sigmoid(model(xb.to(device), attention_mask.to(device))).cpu().numpy()
                for row, chunk in enumerate(batch):
                    left = chunk["start"]
                    n = len(chunk["y"])
                    probs[left:left + n] += scores[row, :n]
                    counts[left:left + n] += 1
            predictions[i] = probs / np.maximum(counts, 1)
    return predictions


def binary_labels(record):
    y = np.zeros(len(record["sequence"]), dtype=np.uint8)
    y[np.asarray(record["sites"], dtype=int) - 1] = 1
    return y


def tolerant_labels(y, radius):
    if radius == 0:
        return y
    out = np.zeros_like(y)
    for position in np.flatnonzero(y):
        out[max(0, position - radius):min(len(y), position + radius + 1)] = 1
    return out


def topk_recovery(records, indices, predictions):
    per_sequence = []
    for i in indices:
        scores = predictions[i]
        selected = []
        for pos in np.argsort(scores)[::-1]:
            if all(abs(int(pos) - prev) > TOP_K_TOLERANCE for prev in selected):
                selected.append(int(pos))
                if len(selected) == TOP_K:
                    break
        unmatched = set(int(p) - 1 for p in records[i]["sites"])
        hits = 0
        for pred in selected:
            candidates = [p for p in unmatched if abs(p - pred) <= TOP_K_TOLERANCE]
            if candidates:
                unmatched.remove(min(candidates, key=lambda p: abs(p - pred)))
                hits += 1
        per_sequence.append(hits / len(records[i]["sites"]))
    return float(np.mean(per_sequence))


def score_predictions(records, indices, predictions):
    y = np.concatenate([binary_labels(records[i]) for i in indices])
    p = np.concatenate([predictions[i] for i in indices])
    if np.unique(y).size != 2:
        raise ValueError("Each held-out group must contain positive and negative residues")
    result = {
        "roc_auc": roc_auc_score(y, p),
        "ap_strict": average_precision_score(y, p),
        "ap_tolerance_2": average_precision_score(np.concatenate([tolerant_labels(binary_labels(records[i]), 2) for i in indices]), p),
        "ap_tolerance_5": average_precision_score(np.concatenate([tolerant_labels(binary_labels(records[i]), 5) for i in indices]), p),
        "top10_site_recovery": topk_recovery(records, indices, predictions),
        "n_test_sequences": len(indices),
        "n_test_positives": int(y.sum()),
        "test_positive_rate": float(y.mean()),
        "n_test_residues": len(y),
    }
    return result


def save_figure(summary: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, architecture in zip(axes, ("ffn", "transformer")):
        frame = summary[summary["architecture"] == architecture]
        for source, label in (("esm3di", "ESM3Di"), ("esmpp_base", "Stock ESM++")):
            rows = frame[frame["embedding_source"] == source].set_index("heldout_group").reindex(sorted(frame["heldout_group"].unique()))
            ax.plot(rows.index, rows["ap_strict_mean"], marker="o", label=label)
        ax.set_title(architecture.upper())
        ax.set_ylabel("Strict average precision")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Held-out cut-site AP by group and embedding source")
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_ap_by_group.png", dpi=220)
    fig.savefig(output_dir / "figure2_ap_by_group.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/polyprotein_cut_esm3di_vs_esmpp"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "esm3di_vs_esmpp")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", type=lambda v: tuple(int(x) for x in v.split(",")),
                        default=DEFAULT_SEEDS, help="comma-separated seeds (default: 17)")
    parser.add_argument("--adapter", type=str, default=None,
                        help="HF PEFT adapter dir/repo to use instead of --checkpoint "
                             "(e.g. the hf_compatible folder of cactuskid13/ESM3di_Small_MLM_3di)")
    parser.add_argument("--architectures", type=lambda v: tuple(v.split(",")),
                        default=("ffn", "transformer"))
    args = parser.parse_args()
    if args.adapter is None and not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_records()
    print("Dataset groups:", dict(Counter(r["group"] for r in records)), flush=True)
    print("Annotated proteins:", len(records), "residues:", sum(len(r["sequence"]) for r in records),
          "cut sites:", sum(len(r["sites"]) for r in records), "device:", device, flush=True)

    features = {}
    for source in ("esm3di", "esmpp_base"):
        features[source] = load_or_extract_embeddings(records, source, args.checkpoint, args.cache_dir, device, args.embedding_batch_size, args.adapter)

    offsets = np.cumsum([0] + [len(r["sequence"]) for r in records])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "seed_metrics.partial.csv"
    prediction_path = args.output_dir / "oof_ffn_predictions.csv"
    seed_rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {(r["heldout_group"], r["embedding_source"], r["architecture"], int(r["seed"])) for r in seed_rows}
    prediction_completed = set()
    if prediction_path.exists():
        prediction_rows = pd.read_csv(prediction_path, usecols=["heldout_group", "embedding_source", "seed"])
        prediction_completed = set(map(tuple, prediction_rows.drop_duplicates().itertuples(index=False, name=None)))
        predictions = pd.read_csv(prediction_path)
        keys = ["heldout_group", "embedding_source", "seed", "sequence_id", "position"]
        predictions = predictions.drop_duplicates(keys, keep="last")
        predictions.to_csv(prediction_path, index=False)
    for group in sorted({r["group"] for r in records}):
        test = [i for i, r in enumerate(records) if r["group"] == group]
        for source in ("esm3di", "esmpp_base"):
            for architecture in args.architectures:
                for seed in args.seeds:
                    key = (group, source, architecture, seed)
                    needs_predictions = architecture == "ffn" and (group, source, seed) not in prediction_completed
                    if key in completed and not needs_predictions:
                        print(f"Reusing completed result: {group} {source} {architecture} seed={seed}", flush=True)
                        continue
                    t0 = time.time()
                    predictions = train_predict(records, features[source], offsets, {group}, architecture, seed, device, args.epochs)
                    metrics = score_predictions(records, test, predictions)
                    row = {"heldout_group": group, "embedding_source": source,
                           "architecture": architecture, "seed": seed, **metrics}
                    seed_rows = [r for r in seed_rows if (r["heldout_group"], r["embedding_source"], r["architecture"], int(r["seed"])) != key]
                    seed_rows.append(row)
                    completed.add(key)
                    pd.DataFrame(seed_rows).to_csv(partial_path, index=False)
                    if architecture == "ffn":
                        prediction_rows = pd.DataFrame([
                            {"heldout_group": group, "embedding_source": source, "seed": seed,
                             "sequence_id": records[i]["identifier"], "position": position + 1,
                             "label": int(label), "score": float(score)}
                            for i in test
                            for position, (label, score) in enumerate(zip(binary_labels(records[i]), predictions[i]))
                        ])
                        prediction_rows.to_csv(prediction_path, mode="a", index=False, header=not prediction_path.exists())
                        prediction_completed.add((group, source, seed))
                    print(f"{group} {source} {architecture} seed={seed}: AP={metrics['ap_strict']:.4f}, "
                          f"AP±5={metrics['ap_tolerance_5']:.4f}, top10={metrics['top10_site_recovery']:.3f} "
                          f"({time.time() - t0:.1f}s)", flush=True)
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    metric_columns = ["roc_auc", "ap_strict", "ap_tolerance_2", "ap_tolerance_5", "top10_site_recovery"]
    summary = seeds.groupby(["heldout_group", "embedding_source", "architecture"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metric_columns},
        **{f"{m}_sd": (m, "std") for m in metric_columns},
        n_test_sequences=("n_test_sequences", "first"),
        n_test_positives=("n_test_positives", "first"),
        test_positive_rate=("test_positive_rate", "first"),
        n_test_residues=("n_test_residues", "first"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    save_figure(summary, args.output_dir)
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "checkpoint": (args.adapter or str(args.checkpoint.resolve())),
        "checkpoint_sha256": (None if args.adapter else digest_file(args.checkpoint)),
        "model": MODEL_NAME, "revision": MODEL_REVISION, "seeds": list(args.seeds),
        "window_overlap": WINDOW_OVERLAP, "max_model_tokens": 1024,
        "tolerance_ap": "residue-level AP after dilating each positive label by +/- radius",
        "top10_site_recovery": "macro mean per-protein recall among top 10 scores after +/-5 residue nonmaximum suppression; one-to-one +/-5 matching",
        "unannotated_sequences": "excluded; only ICTV records with parsed cleavage annotations are evaluated",
        "outer_split": "leave one of four viral groups out; sequences are unique ICTV protein records",
        "heads": {
            "ffn": {"hidden": [256, 128], "dropout": 0.45, "input_dropout": 0.10,
                    "epochs": args.epochs, "learning_rate": 1e-3, "weight_decay": 3e-4,
                    "oscar_l1": 1e-6, "oscar_pair": 2e-7},
            "transformer": {"d_model": 256, "heads": 8, "layers": 2, "dropout": 0.35,
                             "input_dropout": 0.08, "epochs": args.epochs, "learning_rate": 1e-3,
                             "weight_decay": 3e-4, "oscar_l1": 1e-6, "oscar_pair": 2e-7,
                             "classifier_window": 512, "classifier_overlap": 64},
        },
        "selection": "fixed settings; no outer-test hyperparameter selection",
        "device": device, "threads": args.threads,
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("Wrote", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
