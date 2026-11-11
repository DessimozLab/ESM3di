#!/usr/bin/env python3
"""Train reproducible all-data polyprotein cut-site heads on curated ICTV records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyses.polyprotein_cut.scripts import run_comparison as comparison
from esm3di.inference.cutting import SmallCutSiteFFN, SmallCutSiteTransformer

DATA_DIR = Path(__file__).resolve().parent / "dataset" / "Amarillovirales"
DEFAULT_CHECKPOINT = Path(json.loads(
    (Path(__file__).resolve().parent / "models" / "cutsite_model_bundle_metadata.json").read_text()
)["esm3di_checkpoint"])
ENTRY_FILES = (
    "Hepacivirus-CHH/entries_fasta_hepacivirus.txt",
    "Orthoflavivirus-CHH/entries_fasta_orthoflavivirus.txt",
    "Pestivirus-CHH/entries_fasta_pestivirus.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def swiss_records(path: Path):
    for block in path.read_text(encoding="utf-8", errors="replace").split("//"):
        if block.lstrip().startswith("ID "):
            yield SeqIO.read(StringIO(block.strip() + "\n//"), "swiss")


def cleavage_sites(record) -> list[int]:
    return sorted({
        int(feature.location.start) + 1
        for feature in record.features
        if feature.type == "SITE" and feature.qualifiers.get("note") == "Cleavage"
    })


def load_training_records(data_dir: Path = DATA_DIR) -> tuple[list[dict], list[dict]]:
    """Combine the 86 ICTV records with novel entry records having explicit cut labels."""
    records = comparison.load_records(data_dir)
    by_accession = {record["accession"]: record for record in records}
    by_sequence = {record["sequence"]: record for record in records}
    excluded = []

    for relative_path in ENTRY_FILES:
        path = data_dir / relative_path
        group = Path(relative_path).parent.name
        for source_record in swiss_records(path):
            sequence = str(source_record.seq).upper()
            accession = source_record.id
            sites = cleavage_sites(source_record)
            existing = by_accession.get(accession) or by_sequence.get(sequence)
            if existing is not None:
                if existing["sequence"] != sequence:
                    raise ValueError(
                        f"Entry {accession} conflicts with an existing sequence "
                        f"({existing['identifier']})"
                    )
                continue
            if not sites:
                excluded.append({
                    "accession": accession,
                    "identifier": source_record.name,
                    "group": group,
                    "reason": "no explicit SITE /note=\"Cleavage\" feature",
                    "source_file": str(path.relative_to(data_dir)),
                })
                continue
            if any(site < 1 or site > len(sequence) for site in sites):
                raise ValueError(f"Cleavage site outside sequence bounds for {accession}")

            record = {
                "identifier": source_record.name,
                "accession": accession,
                "group": group,
                "sequence": sequence,
                "sites": sites,
                "source_path": str(path),
                "source_kind": "additional_entry",
                "site_evidence": [
                    {
                        "position": int(feature.location.start) + 1,
                        "evidence": feature.qualifiers.get("evidence", "not supplied"),
                    }
                    for feature in source_record.features
                    if feature.type == "SITE"
                    and feature.qualifiers.get("note") == "Cleavage"
                ],
            }
            records.append(record)
            by_accession[accession] = record
            by_sequence[sequence] = record

    accessions = [record["accession"] for record in records]
    if len(accessions) != len(set(accessions)):
        raise ValueError("Duplicate accessions found in curated training records")
    return records, excluded


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_vector(records: list[dict], offsets: np.ndarray) -> np.ndarray:
    labels = np.zeros(int(offsets[-1]), dtype=np.float32)
    for index, record in enumerate(records):
        labels[offsets[index] + np.asarray(record["sites"], dtype=np.int64) - 1] = 1
    return labels


def train_ffn(features, labels, config: dict, device: torch.device, seed: int):
    set_seed(seed)
    model = SmallCutSiteFFN(
        d_in=960, d_hidden=config["d_hidden"], dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    positive = int(labels.sum())
    positive_weight = torch.tensor((len(labels) - positive) / positive, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    rng = np.random.default_rng(seed)

    for epoch in range(config["epochs"]):
        model.train()
        order = rng.permutation(len(labels))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            xb = torch.as_tensor(np.asarray(features[indices], dtype=np.float32), device=device)
            yb = torch.as_tensor(labels[indices], device=device).unsqueeze(-1)
            xb = torch.nn.functional.dropout(xb, p=config["input_dropout"], training=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        print(f"FFN epoch {epoch + 1}/{config['epochs']}: loss={total_loss / batches:.6f}", flush=True)

    model.cpu().eval()
    return model


def make_transformer_chunks(records: list[dict]) -> list[dict]:
    chunks = []
    for index, record in enumerate(records):
        length = len(record["sequence"])
        start = 0
        while start < length:
            end = min(start + 512, length)
            labels = np.zeros(end - start, dtype=np.float32)
            for site in record["sites"]:
                if start < site <= end:
                    labels[site - start - 1] = 1
            valid = np.ones(end - start, dtype=np.float32)
            if start:
                valid[:64] = 0
            chunks.append({
                "record_index": index,
                "start": start,
                "end": end,
                "y": labels,
                "valid": valid,
            })
            if end == length:
                break
            start = end - 64
    return chunks


def batch_transformer_chunks(chunks: list[dict], features, offsets: np.ndarray):
    batch = []
    for chunk in chunks:
        record_index = chunk["record_index"]
        left = int(offsets[record_index]) + chunk["start"]
        right = int(offsets[record_index]) + chunk["end"]
        batch.append({
            **chunk,
            "X": np.asarray(features[left:right], dtype=np.float32),
        })
    return comparison.batch_chunks(batch)


def train_transformer(
    records: list[dict],
    features,
    offsets: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int,
):
    set_seed(seed)
    model = SmallCutSiteTransformer(
        d_in=960,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    positives = sum(len(record["sites"]) for record in records)
    tokens = int(offsets[-1])
    positive_weight = torch.tensor((tokens - positives) / positives, device=device)
    rng = np.random.default_rng(seed)
    chunks = make_transformer_chunks(records)

    for epoch in range(config["epochs"]):
        model.train()
        order = rng.permutation(len(chunks))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), config["batch_size"]):
            selected = [chunks[index] for index in order[start:start + config["batch_size"]]]
            xb, yb, loss_mask, attention_mask = batch_transformer_chunks(selected, features, offsets)
            xb, yb = xb.to(device), yb.to(device)
            loss_mask, attention_mask = loss_mask.to(device), attention_mask.to(device)
            xb = torch.nn.functional.dropout(xb, p=config["input_dropout"], training=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, attention_mask)
            per_token = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none", pos_weight=positive_weight,
            )
            loss = (per_token * loss_mask).sum() / loss_mask.sum().clamp_min(1)
            loss = loss + comparison.oscar_penalty(
                model.head[-1].weight,
                lambda_l1=config["oscar_l1"],
                lambda_pair=config["oscar_pair"],
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        print(
            f"Transformer epoch {epoch + 1}/{config['epochs']}: "
            f"loss={total_loss / batches:.6f}",
            flush=True,
        )

    model.cpu().eval()
    return model


def checkpoint_metadata(
    architecture: str,
    config: dict,
    records: list[dict],
    labels: np.ndarray,
    checkpoint_path: Path,
    seed: int,
    manifest_hash: str,
) -> dict:
    common = {
        "d_in": 960,
        "model_type": architecture,
        "weight_decay": config["weight_decay"],
        "oscar_l1": config["oscar_l1"],
        "oscar_pair": config["oscar_pair"],
        "train_epochs": config["epochs"],
        "lr": config["learning_rate"],
        "n_sequences": len(records),
        "n_tokens": int(len(labels)),
        "n_positive_tokens": int(labels.sum()),
        "esm3di_checkpoint": str(checkpoint_path.resolve()),
        "seed": seed,
        "training_manifest_sha256": manifest_hash,
    }
    if architecture == "ffn":
        common.update({
            "d_hidden": config["d_hidden"],
            "dropout": config["dropout"],
            "input_dropout": config["input_dropout"],
        })
    else:
        common.update({
            "d_model": config["d_model"],
            "nhead": config["nhead"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"],
            "input_dropout": config["input_dropout"],
            "classifier_window": 512,
            "classifier_overlap": 64,
        })
    return common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("/tmp/polyprotein_cut_ictv_88_embeddings"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "models" / "ictv_88_seed17",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=min(8, torch.get_num_threads()))
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.dry_run and args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new --output-dir: {args.output_dir}"
        )

    records, excluded = load_training_records()
    groups = Counter(record["group"] for record in records)
    total_tokens = sum(len(record["sequence"]) for record in records)
    total_sites = sum(len(record["sites"]) for record in records)
    if args.dry_run:
        print(json.dumps({
            "records": len(records),
            "groups": dict(sorted(groups.items())),
            "residues": total_tokens,
            "positive_sites": total_sites,
            "excluded_unlabelled_entries": len(excluded),
            "excluded_accessions": [row["accession"] for row in excluded],
        }, indent=2))
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Training {len(records)} records from {dict(groups)}; "
        f"{total_tokens} residues, {total_sites} cleavage sites on {device}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = sorted({
        Path(record["source_path"])
        for record in records
    } | {
        DATA_DIR / path for path in ENTRY_FILES
    })
    manifest = {
        "dataset": "curated ICTV cut-site records plus explicitly annotated entry records",
        "label_rule": "UniProt SITE feature with note=Cleavage; use feature start as 1-based cut position",
        "unlabelled_entry_policy": "exclude entries without an explicit cleavage SITE annotation",
        "records": [
            {
                "identifier": record["identifier"],
                "accession": record["accession"],
                "group": record["group"],
                "sequence_length": len(record["sequence"]),
                "sequence_sha256": hashlib.sha256(record["sequence"].encode()).hexdigest(),
                "cleavage_sites_1based": record["sites"],
                "source_file": str(Path(record["source_path"]).relative_to(REPO_ROOT)),
                "source_kind": record.get("source_kind", "ictv"),
                **(
                    {"site_evidence": record["site_evidence"]}
                    if "site_evidence" in record else {}
                ),
            }
            for record in records
        ],
        "excluded_entries": excluded,
        "source_files": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in source_files
        },
        "embedding_checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "counts": {
            "records": len(records),
            "residues": total_tokens,
            "positive_sites": total_sites,
            "groups": dict(sorted(groups.items())),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "device": str(device),
            "ffn": {
                "dropout": 0.45, "input_dropout": 0.10, "weight_decay": 1e-4,
                "oscar_l1": 0.0, "oscar_pair": 0.0,
                "learning_rate": 1e-3, "batch_size": 2048,
            },
            "transformer": {
                "d_model": 256, "nhead": 8, "num_layers": 2, "dropout": 0.35,
                "input_dropout": 0.08, "weight_decay": 3e-4,
                "oscar_l1": 3e-6, "oscar_pair": 6e-7,
                "learning_rate": 1e-3, "batch_size": 4,
                "classifier_window": 512, "classifier_overlap": 64,
            },
        },
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = sha256_file(manifest_path)

    features = comparison.load_or_extract_embeddings(
        records, "esm3di", args.checkpoint, args.cache_dir, str(device),
        args.embedding_batch_size,
    )
    offsets = np.cumsum([0] + [len(record["sequence"]) for record in records])
    labels = label_vector(records, offsets)

    ffn_config = {
        "d_hidden": 256, "dropout": 0.45, "input_dropout": 0.10,
        "weight_decay": 1e-4, "oscar_l1": 0.0, "oscar_pair": 0.0,
        "learning_rate": 1e-3, "batch_size": 2048, "epochs": args.epochs,
    }
    transformer_config = {
        "d_model": 256, "nhead": 8, "num_layers": 2, "dropout": 0.35,
        "input_dropout": 0.08, "weight_decay": 3e-4,
        "oscar_l1": 3e-6, "oscar_pair": 6e-7,
        "learning_rate": 1e-3, "batch_size": 4, "epochs": args.epochs,
    }

    for architecture, trainer, config in (
        ("ffn", train_ffn, ffn_config),
        ("transformer", train_transformer, transformer_config),
    ):
        model = (
            trainer(features, labels, config, device, args.seed)
            if architecture == "ffn"
            else trainer(records, features, offsets, config, device, args.seed)
        )
        checkpoint = checkpoint_metadata(
            architecture, config, records, labels, args.checkpoint, args.seed, manifest_hash,
        )
        checkpoint["state_dict"] = model.state_dict()
        destination = args.output_dir / f"cutsite_{architecture}_ictv_88.pt"
        torch.save(checkpoint, destination)
        print(f"Saved {destination}", flush=True)

    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "training_manifest_sha256": manifest_hash,
        "ffn_checkpoint_sha256": sha256_file(args.output_dir / "cutsite_ffn_ictv_88.pt"),
        "transformer_checkpoint_sha256": sha256_file(args.output_dir / "cutsite_transformer_ictv_88.pt"),
        "n_records": len(records),
        "n_residues": total_tokens,
        "n_positive_sites": total_sites,
        "n_excluded_unlabelled_entries": len(excluded),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
