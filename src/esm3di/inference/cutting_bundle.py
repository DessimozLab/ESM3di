"""Portable safetensors bundles for the existing polyprotein cutter."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import torch


MODEL_ARGUMENTS = (
    "hf_model_name", "hf_model", "revision", "num_labels", "lora_r", "lora_alpha",
    "lora_dropout", "target_modules", "use_cnn_head", "cnn_num_layers",
    "cnn_kernel_size", "cnn_dropout", "use_transformer_head",
    "use_plddt_prediction_head", "plddt_prediction_mode", "plddt_num_bins",
    "transformer_head_dim", "transformer_head_layers", "transformer_head_dropout",
    "transformer_head_num_heads", "use_iterative_transformer_head",
    "iterative_head_max_iterations", "iterative_head_halt_threshold",
    "iterative_head_lambda_p", "iterative_head_prior_p", "use_positional_encoding",
    "use_hidden_state_feedback", "use_gru_gate", "aux_track_num_bins",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bundle(directory: str | Path) -> dict:
    directory = Path(directory)
    config = json.loads((directory / "cutting_config.json").read_text())
    if config["format"] not in {"esm3di-cutting-v1", "esm3di-cutting-v2"}:
        raise ValueError(f"Unsupported cutter bundle format: {config['format']}")
    names = ["ffn.safetensors", "transformer.safetensors"]
    if config["format"] == "esm3di-cutting-v1":
        names.append("embedding.safetensors")
    for name in names:
        if sha256(directory / name) != config["sha256"][name]:
            raise ValueError(f"Cutter bundle checksum mismatch: {directory / name}")
    return config


def resolve_bundle(source: str, subfolder: str, revision: str | None) -> Path:
    if Path(source).is_dir():
        directory = Path(source)
    else:
        relative = Path(subfolder)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Cutter subfolder must be a relative path without '..'")
        root = snapshot_download(
            repo_id=source, revision=revision,
            allow_patterns=[f"{subfolder}/*"] if subfolder else ["*"],
        )
        directory = Path(root) / subfolder
    read_bundle(directory)
    return directory


def load_bundle_checkpoint(directory: Path, component: str) -> dict:
    config = read_bundle(directory)
    if component not in {"embedding", "ffn", "transformer"}:
        raise ValueError(f"Unknown cutter component: {component}")
    metadata = dict(config[component])
    key = "model_state_dict" if component == "embedding" else "state_dict"
    metadata[key] = load_file(str(directory / f"{component}.safetensors"))
    return metadata


def export_bundle(
    embedding: Path, ffn: Path, transformer: Path, output: Path,
    embedding_repo: str | None = None, embedding_revision: str | None = None,
) -> Path:
    """Export trusted local checkpoints without loading models or running inference."""
    from .._core.checkpoints import DEFAULT_HF_MODEL, resolve_hf_revision

    sources = {"embedding": embedding, "ffn": ffn, "transformer": transformer}
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in sources.items()
    }
    for name in ("ffn", "transformer"):
        declared = checkpoints[name].get("esm3di_checkpoint")
        if declared and Path(declared).resolve() != embedding.resolve():
            raise ValueError(f"{name} metadata names a different embedding checkpoint: {declared}")
    original = checkpoints["embedding"]
    args = {key: value for key, value in original.get("args", {}).items() if key in MODEL_ARGUMENTS}
    base = args.get("hf_model_name", args.get("hf_model", DEFAULT_HF_MODEL))
    args["revision"] = resolve_hf_revision(base, args.get("revision"))
    config = {
        "format": "esm3di-cutting-v1",
        "cutoff": 0.5,
        "embedding": {
            "args": args,
            **{key: original[key] for key in (
                "label_vocab", "plddt_label_vocab", "lora_target_modules", "aux_track_num_bins",
            ) if key in original},
        },
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "sha256": {},
    }
    if embedding_repo:
        if not embedding_revision:
            raise ValueError("A heads-only export requires a pinned embedding repository revision")
        config["format"] = "esm3di-cutting-v2"
        config["embedding"] = {
            "repo_id": embedding_repo, "repo_revision": embedding_revision,
            "backbone_revision": "46c5f7d",
        }
    for name in ("ffn", "transformer"):
        config[name] = {
            key: value for key, value in checkpoints[name].items()
            if key not in {"state_dict", "esm3di_checkpoint"}
        }
    output.mkdir(parents=True, exist_ok=False)
    for name, checkpoint in checkpoints.items():
        if name == "embedding" and embedding_repo:
            continue
        key = "model_state_dict" if name == "embedding" else "state_dict"
        tensors = {key: value.detach().cpu().contiguous().clone() for key, value in checkpoint[key].items()}
        destination = output / f"{name}.safetensors"
        save_file(tensors, str(destination))
        config["sha256"][destination.name] = sha256(destination)
    (output / "cutting_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "README.md").write_text(
        "# ESM3Di polyprotein cutter\n\n"
        "Custom ESM3Di safetensors bundle, loaded with `esm3di-infer cut`, not "
        "`transformers.AutoModel.from_pretrained`.\n\n"
        + (
            "Contains only the FFN/Transformer cut-site heads. Reuses the existing "
            "Hub 3Di model pinned in `cutting_config.json`; no backbone is duplicated.\n\n"
            if embedding_repo else
            "Contains the original embedding checkpoint and FFN/Transformer heads. "
            "The base model/tokenizer revision is recorded in `cutting_config.json`.\n\n"
        ) +
        "Historical `esm3di_from_mlm` weights did not receive MLM backbone transfer. "
        "The classifier heads were trained on all 231 labelled sequences; held-out "
        "group average precision was low (FFN 0.0669, Transformer 0.0329). "
        "Predicted cuts are exploratory, not validated biological annotations.\n\n"
        "Upload all files together under `cutter/` at the repository root. "
        "Source and export SHA256 hashes are in `cutting_config.json`. "
        "No optimizer states or machine-specific training paths are included.\n"
    )
    read_bundle(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export trusted local cutter checkpoints for manual Hub upload.")
    parser.add_argument("--esm3di-checkpoint", type=Path, required=True)
    parser.add_argument("--ffn-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--embedding-repo", help="Reuse an existing Hub model; export classifier heads only")
    parser.add_argument("--embedding-revision", help="Immutable Hub commit of the existing embedding model")
    args = parser.parse_args(argv)
    directory = export_bundle(
        args.esm3di_checkpoint, args.ffn_checkpoint, args.transformer_checkpoint, args.output_dir,
        args.embedding_repo, args.embedding_revision,
    )
    print(f"Prepared cutter bundle: {directory}")


if __name__ == "__main__":
    main()
