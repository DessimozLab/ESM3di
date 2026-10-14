"""Checkpoint utilities for the ESM++ MLM -> 3Di pipeline.

This module is the single place that knows how to

* pin a HuggingFace remote-code revision that works with the installed
  ``transformers`` version (:func:`resolve_hf_revision`),
* read backbone weights from an MLM fine-tuning output directory and copy them
  into a freshly loaded HF model *before* LoRA is applied
  (:func:`load_mlm_state_dict`, :func:`transfer_backbone_weights`),
* classify an ``--init-from`` argument (:func:`detect_init_source`), and
* rebuild a trained 3Di tagger from a ``.pt`` checkpoint
  (:func:`load_3di_checkpoint`).

Checkpoint formats
------------------
MLM checkpoint (output of ``esm3di mlm``)
    A HuggingFace model directory: ``config.json``, ``model.safetensors`` (or
    ``pytorch_model.bin``) and tokenizer files. Keys look like
    ``embed.weight``, ``transformer.blocks.N...``, ``sequence_head...``.
    Directories containing only a PEFT adapter (``adapter_config.json``) are
    also accepted; the adapter is merged into its base model on load.

3Di checkpoint (output of ``esm3di train``)
    A ``torch.save`` dict with at least ``model_state_dict``, ``label_vocab``,
    ``mask_label_chars``, ``args``, ``epoch`` and ``optimizer_state_dict``.
    ``model_state_dict`` holds the *complete* model (backbone, LoRA adapters
    and head), so only the architecture (``args['hf_model']``) is needed to
    rebuild it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# Remote-code revisions of the Synthyra ESM++ repos known to load with
# transformers 4.48.x. Newer revisions on the hub import ``fastplms`` and
# ``transformers.AttentionInterface`` which do not exist in 4.48.
KNOWN_HF_REVISIONS: Dict[str, str] = {
    "Synthyra/ESMplusplus_small": "056ceb518bbcf1a69f18d2504e9701513684b846",
    "Synthyra/ESMplusplus_large": "36ef85d43b09001f8006774ff35ae1fa62f06a66",
}

DEFAULT_HF_MODEL = "Synthyra/ESMplusplus_small"

# FoldSeek's 3Di alphabet uses the 20 amino-acid letters.
THREEDI_VOCAB: List[str] = list("ACDEFGHIKLMNPQRSTVWY")

_HEAD_PREFIXES = ("sequence_head", "lm_head", "classifier", "score")


def resolve_hf_revision(hf_model: str, revision: Optional[str] = None) -> Optional[str]:
    """Return the HF revision to use for ``hf_model``.

    Priority: explicit ``revision`` > ``ESM3DI_HF_REVISION`` env var > pinned
    known-good revision > ``None`` (hub default). Pass ``revision="main"`` to
    deliberately use the latest hub code.
    """
    if revision:
        return revision
    env_rev = os.environ.get("ESM3DI_HF_REVISION")
    if env_rev:
        return env_rev
    if os.path.isdir(hf_model):
        return None
    return KNOWN_HF_REVISIONS.get(hf_model)


# ---------------------------------------------------------------------------
# init-source detection
# ---------------------------------------------------------------------------

def is_3di_checkpoint(path: str) -> bool:
    return os.path.isfile(path) and path.endswith((".pt", ".pth"))


def is_mlm_checkpoint_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    return "config.json" in names or "adapter_config.json" in names


def detect_init_source(init_from: Optional[str]) -> str:
    """Classify ``--init-from``: ``'base'``, ``'mlm'`` or ``'3di'``."""
    if not init_from:
        return "base"
    if is_3di_checkpoint(init_from):
        return "3di"
    if os.path.isdir(init_from):
        if is_mlm_checkpoint_dir(init_from):
            return "mlm"
        raise ValueError(
            f"--init-from directory {init_from!r} does not look like an MLM checkpoint "
            "(expected config.json + model.safetensors, or a PEFT adapter). If this is an "
            "MLM output directory, point at one of its sub-directories, e.g. "
            f"{os.path.join(init_from, 'best')} or {os.path.join(init_from, 'final')}."
        )
    if os.path.exists(init_from):
        raise ValueError(f"Unrecognised --init-from path: {init_from!r}")
    # Anything else is treated as a HuggingFace hub id.
    return "base"


def mlm_base_model_name(mlm_dir: str) -> Optional[str]:
    """Best-effort lookup of the HF base model an MLM directory was trained from."""
    meta_path = os.path.join(mlm_dir, "esm3di_mlm.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        if meta.get("base_model"):
            return meta["base_model"]
    for fname in ("adapter_config.json", "config.json"):
        fpath = os.path.join(mlm_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath) as fh:
                cfg = json.load(fh)
            name = cfg.get("base_model_name_or_path") or cfg.get("_name_or_path")
            if name and not os.path.isdir(name):
                return name
    return None


# ---------------------------------------------------------------------------
# MLM -> backbone weight transfer
# ---------------------------------------------------------------------------

def _read_weight_file(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return obj


def _find_weight_file(directory: str) -> Optional[str]:
    for fname in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            return fpath
    return None


def _strip_known_prefixes(key: str) -> str:
    for prefix in ("module.", "base_model.model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key.replace(".base_layer.", ".")


def load_mlm_state_dict(mlm_dir: str, revision: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Return the full (merged) backbone state dict stored in an MLM directory.

    Handles full-weight HF directories and PEFT adapter directories.
    """
    if os.path.isfile(os.path.join(mlm_dir, "adapter_config.json")) and _find_weight_file(mlm_dir) is None:
        from peft import PeftModel
        from transformers import AutoModelForMaskedLM

        with open(os.path.join(mlm_dir, "adapter_config.json")) as fh:
            base_name = json.load(fh)["base_model_name_or_path"]
        base = AutoModelForMaskedLM.from_pretrained(
            base_name, trust_remote_code=True, revision=resolve_hf_revision(base_name, revision)
        )
        merged = PeftModel.from_pretrained(base, mlm_dir).merge_and_unload()
        return {k: v.detach().cpu() for k, v in merged.state_dict().items()}

    weight_file = _find_weight_file(mlm_dir)
    if weight_file is None:
        raise FileNotFoundError(
            f"No model weights (model.safetensors / pytorch_model.bin) found in {mlm_dir!r}. "
            "Only config/tokenizer files are present - the weights may have been deleted."
        )
    raw = _read_weight_file(weight_file)
    state = {}
    for key, value in raw.items():
        if "lora_" in key:
            raise ValueError(
                f"{weight_file} contains unmerged LoRA weights ({key}); export a merged model first."
            )
        state[_strip_known_prefixes(key)] = value
    return state


@dataclass
class TransferReport:
    copied: int
    partially_copied: List[str]
    missing: List[str]
    skipped_heads: int

    def summary(self) -> str:
        msg = f"copied {self.copied} backbone tensors"
        if self.partially_copied:
            msg += f", row-sliced {len(self.partially_copied)} ({', '.join(self.partially_copied)})"
        if self.missing:
            msg += f", {len(self.missing)} backbone tensors NOT found in MLM checkpoint"
        return msg


def transfer_backbone_weights(
    target_model: torch.nn.Module,
    source_state: Dict[str, torch.Tensor],
    min_fraction: float = 0.95,
    include_heads: bool = False,
) -> TransferReport:
    """Copy backbone weights from ``source_state`` into ``target_model`` in place.

    ``target_model`` must be the *plain* HF model (before LoRA wrapping). Head
    weights (``sequence_head``/``classifier``) are skipped. Embedding matrices
    whose vocabulary was resized during MLM training (e.g. 64 -> 33 rows) are
    copied row-wise for the overlapping token ids.

    Raises ``RuntimeError`` if fewer than ``min_fraction`` of the target's
    backbone tensors could be matched - this guards against silently training
    from the stock weights.
    """
    target_state = target_model.state_dict()
    backbone_keys = [k for k in target_state if include_heads or not k.startswith(_HEAD_PREFIXES)]
    copied = 0
    partial: List[str] = []
    missing: List[str] = []
    skipped_heads = 0 if include_heads else sum(1 for k in source_state if k.startswith(_HEAD_PREFIXES))

    with torch.no_grad():
        for key in backbone_keys:
            src = source_state.get(key)
            if src is None:
                missing.append(key)
                continue
            dst = target_state[key]
            if src.shape == dst.shape:
                dst.copy_(src.to(dst.dtype))
                copied += 1
            elif (
                src.dim() == dst.dim() >= 1
                and src.shape[1:] == dst.shape[1:]
                and src.shape[0] != dst.shape[0]
            ):
                rows = min(src.shape[0], dst.shape[0])
                dst[:rows].copy_(src[:rows].to(dst.dtype))
                copied += 1
                partial.append(key)
            else:
                missing.append(key)

    if backbone_keys and copied / len(backbone_keys) < min_fraction:
        raise RuntimeError(
            f"Only {copied}/{len(backbone_keys)} backbone tensors matched the MLM checkpoint. "
            f"Is the MLM checkpoint from the same base architecture? Missing e.g.: {missing[:5]}"
        )
    return TransferReport(copied=copied, partially_copied=partial, missing=missing, skipped_heads=skipped_heads)


def load_mlm_model(
    source: str,
    revision: Optional[str] = None,
    base_model: Optional[str] = None,
):
    """Load an ESM++ masked-LM model from a hub id or an MLM checkpoint directory.

    For directories the architecture/code is taken from the *pinned* base model
    revision and the weights (including the LM head) are copied in. This avoids
    the ``auto_map`` in saved ``config.json`` files pulling incompatible
    remote code from the hub's latest revision.
    """
    from transformers import AutoModelForMaskedLM

    if os.path.isdir(source):
        base = base_model or mlm_base_model_name(source)
        if base is None:
            raise ValueError(f"Cannot determine the base model of {source!r}; pass base_model=")
        model = AutoModelForMaskedLM.from_pretrained(
            base, trust_remote_code=True, revision=resolve_hf_revision(base, revision)
        )
        report = transfer_backbone_weights(model, load_mlm_state_dict(source, revision), include_heads=True)
        return model, base, report
    model = AutoModelForMaskedLM.from_pretrained(
        source, trust_remote_code=True, revision=resolve_hf_revision(source, revision)
    )
    return model, source, None


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer(model=None, hf_model: Optional[str] = None, revision: Optional[str] = None):
    """Return a tokenizer for an ESM-2 / ESM++ model.

    ESM++ exposes its tokenizer as ``model.tokenizer`` (``AutoTokenizer`` does not
    know ``EsmSequenceTokenizer``), so prefer that.
    """
    seen = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        tok = getattr(current, "tokenizer", None)
        if tok is not None and not isinstance(tok, torch.nn.Module):
            return tok
        current = getattr(current, "base_model", None) or getattr(current, "model", None)
    if hf_model is None:
        raise ValueError("Cannot determine tokenizer: provide hf_model")
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if os.path.isdir(hf_model) and os.path.isfile(os.path.join(hf_model, "tokenizer.json")):
        try:
            return AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        except Exception:
            return PreTrainedTokenizerFast.from_pretrained(hf_model)
    return AutoTokenizer.from_pretrained(
        hf_model, trust_remote_code=True, revision=resolve_hf_revision(hf_model, revision)
    )


# ---------------------------------------------------------------------------
# 3Di checkpoint loading
# ---------------------------------------------------------------------------

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a ``module.`` prefix added by ``torch.nn.DataParallel``."""
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    return state_dict


def read_3di_checkpoint(path: str, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{path!r} is not an ESM3Di checkpoint (no 'model_state_dict').")
    ckpt["model_state_dict"] = strip_module_prefix(ckpt["model_state_dict"])
    args = ckpt.get("args") or {}
    if not isinstance(args, dict):
        args = vars(args)
    ckpt["args"] = args
    return ckpt


_UNSUPPORTED_LEGACY_FLAGS = ("use_iterative_backbone_head",)


def model_kwargs_from_checkpoint(ckpt: dict) -> dict:
    """Translate a checkpoint's saved ``args`` into ``ESM3DiModel`` kwargs."""
    args = ckpt["args"]
    for flag in _UNSUPPORTED_LEGACY_FLAGS:
        if args.get(flag):
            raise ValueError(
                f"Checkpoint was trained with {flag}=True, which is experimental. "
                "Load it with esm3di.experimental instead."
            )
    hf_model = args.get("hf_model") or DEFAULT_HF_MODEL
    if "t5" in hf_model.lower() or "ankh" in hf_model.lower():
        raise ValueError(f"T5/Ankh checkpoints ({hf_model}) are experimental and not supported here.")

    target_modules = ckpt.get("lora_target_modules") or args.get("lora_target_modules")
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]

    plddt_vocab = ckpt.get("plddt_label_vocab") or []
    kwargs = dict(
        hf_model_name=hf_model,
        num_labels=len(ckpt["label_vocab"]),
        lora_r=args.get("lora_r", 16),
        lora_alpha=args.get("lora_alpha", 32.0),
        lora_dropout=args.get("lora_dropout", 0.01),
        target_modules=target_modules or None,
        use_cnn_head=args.get("use_cnn_head", False),
        cnn_num_layers=args.get("cnn_num_layers", 2),
        cnn_kernel_size=args.get("cnn_kernel_size", 3),
        cnn_dropout=args.get("cnn_dropout", 0.1),
        use_transformer_head=args.get("use_transformer_head", False),
        transformer_head_dim=args.get("transformer_head_dim", 256),
        transformer_head_layers=args.get("transformer_head_layers", 2),
        transformer_head_dropout=args.get("transformer_head_dropout", 0.1),
        transformer_head_num_heads=args.get("transformer_head_num_heads"),
        use_iterative_transformer_head=args.get("use_iterative_transformer_head", False),
        iterative_head_max_iterations=args.get("iterative_head_max_iterations", 5),
        iterative_head_halt_threshold=args.get("iterative_head_halt_threshold", 0.95),
        iterative_head_lambda_p=args.get("iterative_head_lambda_p", 0.01),
        iterative_head_prior_p=args.get("iterative_head_prior_p", 0.5),
        use_positional_encoding=args.get("use_positional_encoding", True),
        use_hidden_state_feedback=args.get("use_hidden_state_feedback", True),
        use_gru_gate=args.get("use_gru_gate", False),
        use_plddt_prediction_head=args.get("use_plddt_prediction_head", False),
        plddt_num_bins=len(plddt_vocab) or 10,
        plddt_prediction_mode=args.get("plddt_prediction_mode", "classification"),
        aux_track_num_bins=ckpt.get("aux_track_num_bins") or args.get("aux_track_num_bins"),
        revision=args.get("hf_revision"),
    )
    return kwargs


# Parameters that some transformers versions create but never use (ESM-2 with
# rotary embeddings); older checkpoints legitimately lack them.
_IGNORABLE_MISSING_SUFFIXES = ("embeddings.position_embeddings.weight",)


def _plain_key(key: str) -> str:
    """Map a PEFT-wrapped parameter name to the underlying dense parameter name."""
    key = key.replace(".base_layer.", ".")
    key = key.replace(".modules_to_save.default.", ".")
    key = key.replace(".original_module.", ".")
    return key


def merge_lora_state_dict(state_dict: Dict[str, torch.Tensor], scaling: float) -> Dict[str, torch.Tensor]:
    """Fold every LoRA delta in ``state_dict`` into dense weights.

    Returns a dict keyed by :func:`_plain_key` names without any ``lora_*``
    entries. For PEFT ``modules_to_save`` wrappers the trained copy
    (``modules_to_save.default``) wins over ``original_module``.
    """
    sd = strip_module_prefix(state_dict)
    merged: Dict[str, torch.Tensor] = {}
    # original_module first so modules_to_save overwrites it
    order = sorted(sd, key=lambda k: 0 if ".original_module." in k else 1)
    for key in order:
        if ".lora_" in key:
            continue
        tensor = sd[key].float() if sd[key].is_floating_point() else sd[key]
        prefix = None
        if key.endswith(".base_layer.weight"):
            prefix = key[: -len(".base_layer.weight")]
        if prefix is not None:
            a, b = f"{prefix}.lora_A.default.weight", f"{prefix}.lora_B.default.weight"
            ea, eb = f"{prefix}.lora_embedding_A.default", f"{prefix}.lora_embedding_B.default"
            if a in sd and b in sd:
                tensor = tensor + (sd[b].float() @ sd[a].float()) * scaling
            elif ea in sd and eb in sd:
                tensor = tensor + (sd[eb].float() @ sd[ea].float()).T * scaling
        merged[_plain_key(key)] = tensor
    return merged


def _load_merged(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], scaling: float) -> None:
    """Load a checkpoint with a foreign LoRA layout by merging it into base weights.

    The checkpoint's LoRA deltas are folded into dense weights, which are copied
    into the current model's base layers; the current model's ``lora_B``
    matrices are zeroed so its adapters start as an exact no-op. Inference is
    therefore identical to the original checkpoint.
    """
    merged = merge_lora_state_dict(state_dict, scaling)
    target = model.state_dict()
    new_state, missing = {}, []
    for key, value in target.items():
        if ".lora_" in key:
            if ".lora_B." in key or ".lora_embedding_B." in key:
                new_state[key] = torch.zeros_like(value)
            continue
        src = merged.get(_plain_key(key))
        if src is None:
            if not key.endswith(_IGNORABLE_MISSING_SUFFIXES):
                missing.append(key)
            continue
        if src.shape != value.shape:
            raise RuntimeError(f"Shape mismatch for {key}: checkpoint {tuple(src.shape)} vs model {tuple(value.shape)}")
        new_state[key] = src.to(value.dtype)
    if missing:
        raise RuntimeError(
            "Checkpoint does not match the model architecture (even after merging LoRA).\n"
            f"  missing ({len(missing)}): {missing[:5]}\n"
            "Check that hf_model / head settings match the checkpoint."
        )
    model.load_state_dict(new_state, strict=False)


def load_state_into_model(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    lora_scaling: Optional[float] = None,
) -> str:
    """Load a 3Di state dict into ``model``.

    Tries an exact load first. If the LoRA layout differs (checkpoints produced
    by older PEFT versions sometimes wrapped extra modules) and ``lora_scaling``
    is given, falls back to :func:`_load_merged`. Returns ``"exact"`` or
    ``"merged"``. Raises ``RuntimeError`` with a readable diff otherwise.
    """
    result = model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    missing = [k for k in result.missing_keys if not k.endswith(_IGNORABLE_MISSING_SUFFIXES)]
    if not missing and not result.unexpected_keys:
        return "exact"
    if lora_scaling is not None:
        _load_merged(model, state_dict, lora_scaling)
        print("  note: checkpoint LoRA layout differs from this PEFT version; "
              "merged its adapters into the base weights (outputs are unchanged).")
        return "merged"
    raise RuntimeError(
        "Checkpoint does not match the model architecture.\n"
        f"  missing ({len(missing)}): {missing[:5]}\n"
        f"  unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}\n"
        "Check that hf_model / head / LoRA settings match the checkpoint."
    )


def lora_scaling_from_checkpoint(ckpt: dict) -> float:
    args = ckpt["args"]
    r = args.get("lora_r", 16)
    alpha = args.get("lora_alpha", 32.0)
    if args.get("use_rslora"):
        return alpha / (r ** 0.5)
    return alpha / r


def load_3di_checkpoint(
    path: str,
    device: Optional[str] = None,
    hf_revision: Optional[str] = None,
    eval_mode: bool = True,
) -> Tuple["ESM3DiModel", dict]:  # noqa: F821
    """Rebuild a trained 3Di tagger from a ``.pt`` checkpoint.

    Returns ``(esm3di_model, checkpoint_dict)``. ``esm3di_model.model`` holds the
    loaded ``nn.Module`` (already moved to ``device``) and
    ``esm3di_model.tokenizer`` the matching tokenizer.
    """
    from .ESM3di_model import ESM3DiModel

    ckpt = read_3di_checkpoint(path)
    kwargs = model_kwargs_from_checkpoint(ckpt)
    if hf_revision:
        kwargs["revision"] = hf_revision
    esm = ESM3DiModel(**kwargs)
    load_state_into_model(esm.model, ckpt["model_state_dict"], lora_scaling=lora_scaling_from_checkpoint(ckpt))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    esm.model.to(device)
    if eval_mode:
        esm.model.eval()
    return esm, ckpt
