import os
import queue
import logging
import tempfile
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification
from transformers import logging as hf_logging
from peft import PeftModel, PeftConfig

from .io import read_fasta, write_fasta
from .model import CNNClassificationHead, ESMWithCNNHead

# Internal logger for this module
logger = logging.getLogger(__name__)

# Suppress default HF head init warnings
hf_logging.set_verbosity_error()

# Resolve physical directory where this file sits: src/esm3di/
PACKAGE_ROOT = Path(__file__).resolve().parent

# Production-grade package defaults
DEFAULT_HF_REPO = PACKAGE_ROOT.parents[1] / "checkpoints" / "hf_compatible"
DEFAULT_BATCH_SIZE = 4
VOCAB_3DI = list("ACDEFGHIKLMNPQRSTVWY")


class ESM3DiPredictor:
    """Production wrapper for ESM++ with fine-tuned LoRA adapters and a custom CNN classification head.

    This class serves as the main high-level inference interface. It configures the
    underlying token classification backbone, applies parameter-efficient adapters,
    binds the custom convolutional classification layers, and handles batch predictions
    both in-memory and via streaming processes.
    """

    def __init__(
            self,
            model_checkpoint_path: Union[str, Path] = DEFAULT_HF_REPO,
            device: Optional[str] = None
    ):
        """Initializes the predictor, builds the model hierarchy, and maps saved weights.

        Args:
            model_checkpoint_path: Folder containing adapter configurations, tokenizers,
                and custom model checkpoint weights.
            device: Runtime target device (e.g., 'cuda', 'cpu', 'mps'). If None,
                it is automatically selected based on hardware support.

        Raises:
            FileNotFoundError: If the provided configuration directory is invalid.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_checkpoint_path = str(model_checkpoint_path)

        self._initialize_model_backbone()

        # Build lookup translation map
        self.idx2char = {i: c for i, c in enumerate(VOCAB_3DI)}

        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else self.device.type.upper()
        )
        logger.info(f"Predictor safely initialized on target device: {self.device} ({device_name})")

    def _initialize_model_backbone(self) -> None:
        """Internal helper logic to reconstruct and load the composite model structure."""
        if not Path(self.model_checkpoint_path).exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.model_checkpoint_path}")

        # 1. Read base model configuration
        logger.debug(f"Loading adapter configuration from: {self.model_checkpoint_path}")
        peft_config = PeftConfig.from_pretrained(self.model_checkpoint_path)
        base_model_name = peft_config.base_model_name_or_path

        # 2. Initialize base model
        logger.debug(f"Loading base structural backbone: {base_model_name}")
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=len(VOCAB_3DI),
            trust_remote_code=True,
        )
        self.tokenizer = base_model.tokenizer

        # 3. Apply PEFT adapters
        logger.debug("Applying PEFT adapter weights...")
        peft_model = PeftModel.from_pretrained(base_model, self.model_checkpoint_path)

        # 4. Instantiate custom CNN classification head
        hidden_size = base_model.config.hidden_size
        cnn_head = CNNClassificationHead(hidden_size=hidden_size)

        # 5. Wrap model components
        self.model = ESMWithCNNHead(peft_model=peft_model, cnn_head=cnn_head)

        # 6. Load custom CNN weights if present
        cnn_weights_path = Path(self.model_checkpoint_path) / "cnn_head.bin"
        if cnn_weights_path.exists():
            logger.debug("Loading saved CNN head weights...")
            self.model.cnn_head.load_state_dict(
                torch.load(cnn_weights_path, map_location=self.device)
            )
        else:
            logger.warning(
                "Missing 'cnn_head.bin' in HF folder! "
                "Inference will run on untrained/random CNN weights."
            )

        # Move to device and set evaluation mode
        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(
            cls,
            model_checkpoint_path: Union[str, Path] = DEFAULT_HF_REPO,
            device: Optional[str] = None
    ) -> "ESM3DiPredictor":
        """Syntactic sugar wrapper matching Hugging Face hub conventions.

        Args:
            model_checkpoint_path: Folder holding configurations and checkpoint models.
            device: Runtime target device.

        Returns:
            An instantiated ESM3DiPredictor pipeline.
        """
        return cls(model_checkpoint_path=model_checkpoint_path, device=device)

    # =========================================================================
    # CORE INFERENCE INTERFACES
    # =========================================================================

    def predict(self, sequence: str) -> str:
        """Predicts the structural 3Di token string for an individual raw amino acid sequence.

        Args:
            sequence: Raw single-letter code amino acid string (e.g., "MKKV...").

        Returns:
            A decoded string of predicted 3Di tokens matching the input length.
        """
        return self.predict_batch([sequence], batch_size=1)[0]

    def predict_batch(self, sequences: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[str]:
        """Predicts structural 3Di token characters across an in-memory batch list of sequences.

        Args:
            sequences: List of raw amino acid strings.
            batch_size: Number of sequences to forward-pass simultaneously.

        Returns:
            A list of predicted 3Di sequences in identical order.
        """
        predicted_strings: List[str] = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(sequences), batch_size),
                desc="Generating 3Di tokens",
                unit="batch",
                leave=False
            ):
                batch_seqs = sequences[i: i + batch_size]

                # Tokenize raw sequence batch
                enc = self.tokenizer(
                    list(batch_seqs),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    add_special_tokens=True,
                    return_special_tokens_mask=True
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                special_mask = enc["special_tokens_mask"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_labels = torch.argmax(outputs.logits, dim=-1)

                # Map token indexes back to vocabulary characters, ignoring special tokens
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
        """Streaming pipeline endpoint for processing input FASTA structures.

        Args:
            input_fasta_path: Path to input amino acid FASTA file.
            output_fasta_path: Destination path where predicted 3Di FASTA will be saved.
            batch_size: Execution batch sizing.
            num_gpus: Count of local GPUs to deploy. Auto-detected if None.
        """
        input_fasta_path = Path(input_fasta_path)
        output_fasta_path = Path(output_fasta_path)

        resolved_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus

        # Delegate execution if multi-GPU distributed options are present
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
    ) -> None:
        """Calculates prediction perplexity streams per sequence position and saves to a TSV file.

        Perplexity tracks structural prediction confidence across sequence coordinates.

        Args:
            input_fasta_path: Path to input amino acid FASTA file.
            output_tsv_path: Path to the tab-separated outputs file.
            batch_size: Execution batch sizing.
        """
        input_fasta_path = Path(input_fasta_path)
        output_tsv_path = Path(output_tsv_path)
        aa_records = read_fasta(str(input_fasta_path))

        with open(output_tsv_path, 'w') as out_f:
            out_f.write("sequence_id\tposition\taa\tperplexity\n")

            with torch.no_grad():
                for i in tqdm(
                    range(0, len(aa_records), batch_size),
                    desc="Computing position perplexity",
                    unit="batch",
                    leave=False
                ):
                    headers, raw_seqs = zip(*aa_records[i: i + batch_size])

                    enc = self.tokenizer(
                        list(raw_seqs),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=True,
                        return_special_tokens_mask=True
                    )

                    logits = self.model(
                        input_ids=enc["input_ids"].to(self.device),
                        attention_mask=enc["attention_mask"].to(self.device)
                    ).logits

                    probs = torch.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Compute Shannon entropy metrics tracking structural perplexity indices
                    entropy = -(probs * log_probs).sum(dim=-1)
                    token_perplexity = torch.exp(entropy)

                    # Align token perplexity metrics back to real input residues
                    for j, (header, raw_seq) in enumerate(zip(headers, raw_seqs)):
                        k = 0
                        for pos in range(token_perplexity.shape[1]):
                            if enc["special_tokens_mask"][j, pos] == 0 and k < len(raw_seq):
                                val = float(token_perplexity[j, pos].item())
                                out_f.write(f"{header}\t{k + 1}\t{raw_seq[k]}\t{val:.6f}\n")
                                k += 1


# =========================================================================
# ASYNC WORKERS FOR COMBINED HORIZONTAL SCALE ARRAYS
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
                raise RuntimeError("Distributed worker context crashed tracking inference metrics.")

        for p in processes:
            p.join()

        _merge_fasta_outputs(shard_outputs, output_3di_fasta, [rec.id for rec in SeqIO.parse(input_fasta, "fasta")])
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)