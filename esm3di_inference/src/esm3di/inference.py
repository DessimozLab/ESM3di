"""Inference engine and prediction interface for ESM3Di structural token generation."""

import os
import sys
import queue
import logging
import tempfile
import shutil
import warnings
import contextlib
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Any

# Mute third-party library noise early during import
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging
from peft import PeftModel, PeftConfig

# Silence Hugging Face and PyTorch backend chatter
hf_logging.set_verbosity_error()
logging.getLogger("torch.nn.attention").setLevel(logging.ERROR)

from .io import read_fasta, write_fasta
from .model import CNNClassificationHead, ESMWithCNNHead

# Module logger
logger = logging.getLogger("esm3di")

# Path resolution defaults
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_HF_REPO = PACKAGE_ROOT.parents[1] / "checkpoints" / "hf_compatible"
DEFAULT_BATCH_SIZE = 4
DEFAULT_REVISION = "46c5f7d"
VOCAB_3DI = list("ACDEFGHIKLMNPQRSTVWY")


class ESM3DiPredictor:
    """High-level predictor interface for ESM3Di sequence-to-3Di translation."""

    def __init__(
        self,
        model_checkpoint_path: Union[str, Path] = DEFAULT_HF_REPO,
        revision: Optional[str] = DEFAULT_REVISION,
        device: Optional[str] = None
    ):
        """Initializes the predictor, loads model components, and registers runtime device."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_checkpoint_path = str(model_checkpoint_path)
        self.revision = revision

        self._initialize_model_backbone()

        self.idx2char = {i: c for i, c in enumerate(VOCAB_3DI)}
        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else self.device.type.upper()
        )
        logger.info(f"Using device: {self.device} ({device_name})")

    def _initialize_model_backbone(self) -> None:
        """Internal worker to construct model hierarchy and load trained weights."""
        if not Path(self.model_checkpoint_path).exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.model_checkpoint_path}")

        # Redirect standard output to suppress implicit model instantiation prints
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            peft_config = PeftConfig.from_pretrained(self.model_checkpoint_path)
            base_model_name = peft_config.base_model_name_or_path

            base_model = AutoModel.from_pretrained(
                base_model_name,
                num_labels=len(VOCAB_3DI),
                trust_remote_code=True,
                revision=self.revision
            )

            if hasattr(base_model, "tokenizer"):
                self.tokenizer = base_model.tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    revision=self.revision,
                )

            peft_model = PeftModel.from_pretrained(base_model, self.model_checkpoint_path)
            hidden_size = base_model.config.hidden_size
            cnn_head = CNNClassificationHead(hidden_size=hidden_size)

            self.model = ESMWithCNNHead(peft_model=peft_model, cnn_head=cnn_head)

            cnn_weights_path = Path(self.model_checkpoint_path) / "cnn_head.bin"
            if cnn_weights_path.exists():
                self.model.cnn_head.load_state_dict(
                    torch.load(cnn_weights_path, map_location=self.device)
                )
            else:
                logger.warning("Missing 'cnn_head.bin' in checkpoint folder. Running with uninitialized CNN weights.")

        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls,
        model_checkpoint_path: Union[str, Path] = DEFAULT_HF_REPO,
        revision: Optional[str] = DEFAULT_REVISION,
        device: Optional[str] = None
    ) -> "ESM3DiPredictor":
        """Factory method to construct an ESM3DiPredictor instance from local or HF weights."""
        logger.info(f"Loading model from: {model_checkpoint_path}")
        return cls(model_checkpoint_path=model_checkpoint_path, revision=revision, device=device)

    # =========================================================================
    # CORE INFERENCE METHODS
    # =========================================================================

    def predict(self, sequence: str) -> str:
        """Predicts 3Di tokens for a single raw amino acid sequence string."""
        res = self.predict_batch([sequence], batch_size=1)[0]
        sys.stdout.flush()
        return res

    def predict_batch(self, sequences: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[str]:
        """Predicts 3Di structural tokens for a list of in-memory protein sequences."""
        predicted_strings: List[str] = []
        disable_pbar = len(sequences) <= 1

        with torch.no_grad():
            for i in tqdm(
                range(0, len(sequences), batch_size),
                disable=disable_pbar,
                desc="Generating 3Di tokens",
                unit="batch",
                leave=False
            ):
                batch_seqs = sequences[i: i + batch_size]

                enc = self.tokenizer(
                    list(batch_seqs),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                    return_special_tokens_mask=True
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                special_mask = enc["special_tokens_mask"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_labels = torch.argmax(outputs.logits, dim=-1)

                for j, raw_seq in enumerate(batch_seqs):
                    pred_3di = []
                    k = 0
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
    ) -> None:
        """Processes an input amino acid FASTA file and writes predicted 3Di outputs to FASTA."""
        input_fasta_path = Path(input_fasta_path)
        output_fasta_path = Path(output_fasta_path)

        resolved_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus

        if resolved_gpus > 1:
            success = _run_multi_gpu_inference(
                str(input_fasta_path), str(output_fasta_path),
                self.model_checkpoint_path, resolved_gpus, batch_size=batch_size
            )
            if success:
                logger.info(f"Saved predicted 3Di sequences to: {output_fasta_path}")
                return

        logger.info(f"Reading sequence inputs from: {input_fasta_path}")
        aa_records = read_fasta(str(input_fasta_path))

        output_fasta_path.parent.mkdir(parents=True, exist_ok=True)
        if not aa_records:
            write_fasta([], str(output_fasta_path))
            logger.info(f"Saved predicted 3Di sequences to: {output_fasta_path}")
            return

        headers, raw_seqs = zip(*aa_records)
        predicted_3dis = self.predict_batch(list(raw_seqs), batch_size=batch_size)

        write_fasta(list(zip(headers, predicted_3dis)), str(output_fasta_path))
        logger.info(f"Saved predicted 3Di sequences to: {output_fasta_path}")

    def output_per_position_perplexity(
        self,
        input_fasta_path: Union[str, Path],
        output_tsv_path: Union[str, Path],
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """Calculates residue-level prediction perplexities and exports them to a TSV file."""
        input_fasta_path = Path(input_fasta_path)
        output_tsv_path = Path(output_tsv_path)

        logger.info(f"Reading sequence inputs from: {input_fasta_path}")
        aa_records = read_fasta(str(input_fasta_path))

        output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_tsv_path, 'w') as out_f:
            out_f.write("sequence_id\tposition\taa\tperplexity\n")

            disable_pbar = len(aa_records) <= 1
            with torch.no_grad():
                for i in tqdm(
                    range(0, len(aa_records), batch_size),
                    disable=disable_pbar,
                    desc="Computing perplexity",
                    unit="batch",
                    leave=False
                ):
                    headers, raw_seqs = zip(*aa_records[i: i + batch_size])

                    enc = self.tokenizer(
                        list(raw_seqs),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        return_special_tokens_mask=True
                    )

                    logits = self.model(
                        input_ids=enc["input_ids"].to(self.device),
                        attention_mask=enc["attention_mask"].to(self.device)
                    ).logits

                    probs = torch.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)

                    entropy = -(probs * log_probs).sum(dim=-1)
                    token_perplexity = torch.exp(entropy)

                    for j, (header, raw_seq) in enumerate(zip(headers, raw_seqs)):
                        k = 0
                        for pos in range(token_perplexity.shape[1]):
                            if enc["special_tokens_mask"][j, pos] == 0 and k < len(raw_seq):
                                val = float(token_perplexity[j, pos].item())
                                out_f.write(f"{header}\t{k + 1}\t{raw_seq[k]}\t{val:.6f}\n")
                                k += 1

        logger.info(f"Saved perplexity metrics to: {output_tsv_path}")


# =========================================================================
# MULTI-GPU DISTRIBUTED INFERENCE WORKERS
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
            if error_event.is_set():
                raise RuntimeError("Worker process failed during parallel inference execution.")

        for p in processes:
            p.join()

        _merge_fasta_outputs(shard_outputs, output_3di_fasta, [rec.id for rec in SeqIO.parse(input_fasta, "fasta")])
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)