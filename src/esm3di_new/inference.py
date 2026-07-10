import os
import queue
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from peft import get_peft_model, LoraConfig

from .io import read_fasta, write_fasta
from .model import ESM3DiModel

# Configure logging framework
logger = logging.getLogger(__name__)

# Module-level package defaults
DEFAULT_HF_REPO = "cactuskid13/esm2small_3di"
DEFAULT_FILENAME = "epoch_3.pt"
DEFAULT_BATCH_SIZE = 4


class ESM3DiPredictor:
    """Wraps an ESM3DiModel to handle data batching and 3Di coordinate prediction loops."""

    def __init__(
            self,
            model_checkpoint_path: Union[str, Path] = DEFAULT_HF_REPO,
            device: Optional[str] = None
    ):
        """Initializes the predictor by resolving, downloading, and loading the model assets.

        Args:
            model_checkpoint_path: Local file path (.pt) or a remote Hugging Face
                repository ID string. Defaults to "cactuskid13/esm2small_3di".
            device: Explicit device string allocation (e.g., "cuda", "cpu"). If None,
                automatically switches to an available CUDA execution context.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_checkpoint_path = model_checkpoint_path

        # 1. Resolve storage pathways and download files from HF Hub if required
        resolved_path = self._resolve_checkpoint_path(model_checkpoint_path)

        # 2. Extract and parse checkpoint state maps
        checkpoint_dict = torch.load(resolved_path, map_location="cpu")
        model_kwargs, args, label_vocab = self._parse_checkpoint_metadata(checkpoint_dict)

        # 3. Instantiate underlying architecture and bind PEFT adapters
        model = self._build_peft_model(model_kwargs, args)

        # 4. Inject and validate weight allocations
        self._load_and_validate_weights(model, checkpoint_dict)

        # 5. Commit components to instance state
        self.model = model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_kwargs["hf_model_name"], trust_remote_code=True)
        self.label_vocab = label_vocab
        self.idx2char = {i: c for i, c in enumerate(label_vocab)}

        logger.info(f"✓ Predictor engine safely launched on device: {self.device}")

    @classmethod
    def from_pretrained(
            cls,
            model_checkpoint_path: Union[str, Path] = DEFAULT_HF_REPO,
            device: Optional[str] = None
    ) -> "ESM3DiPredictor":
        """Syntactic sugar wrapper matching Hugging Face conventions.

        Delegates initialization arguments straight down to the main constructor.
        """
        return cls(model_checkpoint_path=model_checkpoint_path, device=device)

    # =========================================================================
    # INTERNAL ASSET STORAGE & LOADING UTILITIES
    # =========================================================================

    @staticmethod
    def _resolve_checkpoint_path(path: Union[str, Path]) -> Path:
        """Determines if a target path is local or must be resolved against remote repositories."""
        if Path(path).exists() and Path(path).is_file():
            return Path(path)

        logger.info(f"Checking remote cache framework for target repository entry: {path}...")
        resolved = hf_hub_download(repo_id=str(path), filename=DEFAULT_FILENAME)
        return Path(resolved)

    @staticmethod
    def _parse_checkpoint_metadata(checkpoint_dict: Any) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """Extracts configuration metadata dictionaries from safe checkpoint payloads."""
        is_dict = isinstance(checkpoint_dict, dict)
        args = checkpoint_dict.get('args', {}) if is_dict else {}
        label_vocab = checkpoint_dict.get('label_vocab', list("ACDEFGHIKLMNPQRSTVWY")) if is_dict else list(
            "ACDEFGHIKLMNPQRSTVWY")
        hf_model_name = args.get('hf_model_name', args.get('hf_model', 'facebook/esm2_t12_35M_UR50D'))

        model_kwargs = { # TODO !!! Complete this if needed !!!
            "hf_model_name": hf_model_name,
            "num_labels": len(label_vocab),
            "lora_r": args.get("lora_r", 8),
            "lora_alpha": args.get("lora_alpha", 16.0),
            "cnn_num_layers": args.get("cnn_num_layers", 2),
            "cnn_kernel_size": args.get("cnn_kernel_size", 3),
            "transformer_head_dim": args.get("transformer_head_dim", 256),
            "transformer_head_layers": args.get("transformer_head_layers", 2),
            "plddt_prediction_mode": args.get("plddt_prediction_mode", "classification"),
            "plddt_num_bins": args.get("plddt_num_bins", 10),
            **{k: args.get(k, False) for k in [
                "use_cnn_head", "use_transformer_head",
                "use_iterative_transformer_head", "use_plddt_prediction_head"
            ]}
        }
        return model_kwargs, args, label_vocab

    @staticmethod
    def _build_peft_model(model_kwargs: Dict[str, Any], args: Dict[str, Any]) -> torch.nn.Module:
        """Constructs the raw structural model backbone and attaches PEFT adapters."""
        logger.info(f"Instantiating structural layout for backbone: {model_kwargs['hf_model_name']}")
        raw_custom_model = ESM3DiModel(**model_kwargs)

        lora_config = LoraConfig(
            task_type=None,
            r=model_kwargs["lora_r"],
            lora_alpha=model_kwargs["lora_alpha"],
            lora_dropout=args.get("lora_dropout", 0.05),
            target_modules=["query", "key", "value", "dense"],
            modules_to_save=["classifier", "plddt_head"]
        )
        return get_peft_model(raw_custom_model, lora_config)

    @staticmethod
    def _load_and_validate_weights(model: torch.nn.Module, checkpoint_dict: Any):
        """Sanitizes legacy DataParallel layouts and loads weights strictly into the model."""
        is_dict = isinstance(checkpoint_dict, dict)
        raw_state_dict = checkpoint_dict.get('model_state_dict', checkpoint_dict) if is_dict else checkpoint_dict

        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            k_clean = k.replace("module.", "", 1)
            if k_clean == "base_model.model.classifier.original_weight":
                k_clean = "base_model.model.classifier.original_module.weight"
            elif k_clean == "base_model.model.classifier.original_bias":
                k_clean = "base_model.model.classifier.original_module.bias"
            clean_state_dict[k_clean] = v

        info = model.load_state_dict(clean_state_dict, strict=False)
        expected_missing = {"base_model.model.esm.embeddings.position_embeddings.weight"}
        actual_missing = set(info.missing_keys) - expected_missing

        if actual_missing or info.unexpected_keys:
            raise RuntimeError(
                f"Critical structural mismatch caught during weight loading!\n"
                f"Missing Keys: {actual_missing}\n"
                f"Unexpected Keys: {info.unexpected_keys}"
            )

    # =========================================================================
    # CORE INFERENCE INTERFACES
    # =========================================================================

    def predict(self, sequence: str) -> str:
        """Predicts the structural 3Di token string for an individual raw amino acid sequence string."""
        return self.predict_batch([sequence], batch_size=1)[0]

    def predict_batch(self, sequences: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[str]:
        """Predicts structural 3Di token characters across an in-memory batch list of sequences."""
        predicted_strings: List[str] = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i: i + batch_size]

                enc = self.tokenizer(
                    list(batch_seqs), return_tensors="pt", padding=True, truncation=True,
                    add_special_tokens=True, return_special_tokens_mask=True
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                special_mask = enc["special_tokens_mask"]

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                pred_labels = torch.argmax(logits, dim=-1)

                for j, raw_seq in enumerate(batch_seqs):
                    pred_3di, k = [], 0
                    for pos in range(pred_labels.shape[1]):
                        if special_mask[j, pos] == 0 and k < len(raw_seq):
                            pred_3di.append(self.idx2char.get(pred_labels[j, pos].item(), 'X'))
                            k += 1
                    predicted_strings.append("".join(pred_3di))

        return predicted_strings

    def predict_fasta(
            self,
            input_fasta_path: Union[str, Path],
            output_fasta_path: Union[str, Path],
            batch_size: int = DEFAULT_BATCH_SIZE,
            num_gpus: Optional[int] = None
    ):
        """Streaming endpoint for FASTA tracking. Automatically branches into parallel workers."""
        input_fasta_path = Path(input_fasta_path)
        output_fasta_path = Path(output_fasta_path)

        resolved_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus

        if resolved_gpus > 1:
            success = _run_multi_gpu_inference(
                str(input_fasta_path), str(output_fasta_path),
                self.model_checkpoint_path, resolved_gpus, batch_size=batch_size
            )
            if success:
                return

        aa_records = read_fasta(str(input_fasta_path))
        if not aa_records:
            write_fasta([], str(output_fasta_path))
            return

        headers, raw_seqs = zip(*aa_records)
        predicted_3dis = self.predict_batch(list(raw_seqs), batch_size=batch_size)
        write_fasta(list(zip(headers, predicted_3dis)), str(output_fasta_path))

    def output_per_position_perplexity(
            self,
            input_fasta_path: Union[str, Path],
            output_tsv_path: Union[str, Path],
            batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """Calculates per-token tracking distributions and saves prediction perplexity streams to a TSV file."""
        input_fasta_path = Path(input_fasta_path)
        output_tsv_path = Path(output_tsv_path)
        aa_records = read_fasta(str(input_fasta_path))

        with open(output_tsv_path, 'w') as out_f:
            out_f.write("sequence_id\tposition\taa\tperplexity\n")
            with torch.no_grad():
                for i in range(0, len(aa_records), batch_size):
                    headers, raw_seqs = zip(*aa_records[i: i + batch_size])
                    enc = self.tokenizer(list(raw_seqs), return_tensors="pt", padding=True, truncation=True,
                                         add_special_tokens=True, return_special_tokens_mask=True)

                    logits = self.model(input_ids=enc["input_ids"].to(self.device),
                                        attention_mask=enc["attention_mask"].to(self.device)).logits
                    probs = torch.exp(F.log_softmax(logits, dim=-1))
                    token_perplexity = torch.exp(-(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1))

                    for j, (header, raw_seq) in enumerate(zip(headers, raw_seqs)):
                        k = 0
                        for pos in range(token_perplexity.shape[1]):
                            if enc["special_tokens_mask"][j, pos] == 0 and k < len(raw_seq):
                                out_f.write(
                                    f"{header}\t{k + 1}\t{raw_seq[k]}\t{float(token_perplexity[j, pos].item()):.6f}\n")
                                k += 1


# =========================================================================
# ISOLATED WORKERS FOR MULTIPROCESSING COMPATIBILITY
# =========================================================================

def _gpu_worker(gpu_id: int, shard_fasta: str, output_fasta: str, checkpoint_path: str, batch_size: int,
                progress_queue: Any, error_event: Any):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        predictor = ESM3DiPredictor(checkpoint_path, device="cuda")
        predictor.predict_fasta(shard_fasta, output_fasta, batch_size=batch_size, num_gpus=1)
        progress_queue.put(("done", gpu_id))
    except Exception as e:
        progress_queue.put(("error", gpu_id, str(e)))
        error_event.set()


def _run_multi_gpu_inference(input_fasta: str, output_3di_fasta: str, checkpoint_path: str, num_gpus: int,
                             batch_size: int = DEFAULT_BATCH_SIZE) -> bool:
    import multiprocessing as mp
    from .preprocessing import _shard_fasta, _merge_fasta_outputs, _count_sequences
    from Bio import SeqIO

    if _count_sequences(input_fasta) <= 1 or num_gpus <= 1:
        return False

    ctx = mp.get_context('spawn')
    temp_dir = tempfile.mkdtemp(prefix="esm3di_shards_")

    try:
        shards = _shard_fasta(input_fasta, num_gpus, temp_dir)
        progress_queue, error_event = ctx.Queue(), ctx.Event()
        shard_outputs, processes = [], []

        for gpu_id, (shard_fasta, _) in enumerate(shards):
            out_path = os.path.join(temp_dir, f"shard_{gpu_id}_3di.fasta")
            shard_outputs.append((shard_fasta, out_path))
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, shard_fasta, out_path, checkpoint_path, batch_size, progress_queue, error_event)
            )
            processes.append(p)
            p.start()

        completed = 0
        while completed < num_gpus:
            try:
                event = progress_queue.get(timeout=0.5)
                if event[0] == "done":
                    completed += 1
                elif event[0] == "error":
                    raise RuntimeError(event[2])
            except queue.Empty:
                pass
            if error_event.is_set(): raise RuntimeError("Worker crashed.")

        for p in processes: p.join()
        _merge_fasta_outputs(shard_outputs, output_3di_fasta, [rec.id for rec in SeqIO.parse(input_fasta, "fasta")])
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)