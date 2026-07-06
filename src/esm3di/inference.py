import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from typing import List, Tuple, Union, Optional
from .io import read_fasta, write_fasta
from .model import ESM3DiModel

# Hugging Face repository and filename
DEFAULT_HF_REPO = "cactuskid13/esm2small_3di"
DEFAULT_FILENAME = "epoch_3.pt"

class ESM3DiPredictor:
    """Wraps an ESM3DiModel to handle fluid data batching and prediction loops.

    Supports both disk-to-disk large scale pipelines and interactive in-memory execution.
    """

    def __init__(self, model_checkpoint_path: str = "default", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Smart Weight Router: Check if we use a local path or require an online download
        if model_checkpoint_path == "default":
            print(f"Using default model. Checking remote cache at {DEFAULT_HF_REPO}...")
            print("Downloading ESM3Di fine-tuned weights...")  # Clear notification for the user

            # Let HF native routines stream the download progress bar safely
            resolved_path = hf_hub_download(
                repo_id=DEFAULT_HF_REPO,
                filename=DEFAULT_FILENAME
            )

        elif os.path.exists(model_checkpoint_path):
            # User provided a direct, valid local path to a .pt file
            resolved_path = model_checkpoint_path
        else:
            # User provided a custom text string, assume it's an alternate Hugging Face Repo ID
            print(f"Local path not found. Attempting Hugging Face Hub download from: {model_checkpoint_path}...")
            resolved_path = hf_hub_download(repo_id=model_checkpoint_path, filename=DEFAULT_FILENAME)

        # Rehydrate model with backward-compatible fallbacks
        checkpoint = torch.load(resolved_path, map_location=self.device)
        args_dict = checkpoint.get('args', {})

        hf_model_name = args_dict.get('hf_model_name', args_dict.get('hf_model', 'facebook/esm2_t33_650M_UR50D'))

        self.model_wrapper = ESM3DiModel(
            hf_model_name=hf_model_name,
            num_labels=len(checkpoint.get('label_vocab', [])),
            **args_dict
        )

        # Load weights cleanly
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
        self.model_wrapper.get_model().load_state_dict(model_state_dict, strict=False)

        self.model = self.model_wrapper.get_model().to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_wrapper.hf_model_name, trust_remote_code=True)
        self.idx2char = {i: c for i, c in enumerate(checkpoint.get("label_vocab", list("ACDEFGHIKLMNPQRSTVWY")))}

    def predict(self, sequence: str) -> str:
        """Predicts the 3Di character sequence for a single raw amino acid string."""
        return self.predict_batch([sequence], batch_size=1)[0]

    def predict_batch(self, sequences: List[str], batch_size: int = 4) -> List[str]:
        """Predicts 3Di character strings for an in-memory list of amino acid strings."""
        predicted_strings = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i + batch_size]

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

    def predict_from_fasta(self, input_fasta_path: str, output_fasta_path: str, batch_size: int = 4):
        """Disk-to-disk streaming interface. Automatically prevents OOMs on massive inputs."""
        aa_records = read_fasta(input_fasta_path)

        # Unpack headers and raw sequences
        headers, raw_seqs = zip(*aa_records) if aa_records else ([], [])

        # Route processing through our highly optimized batch loop
        predicted_3dis = self.predict_batch(list(raw_seqs), batch_size=batch_size)

        # Pack records back up and stream to file
        output_records = list(zip(headers, predicted_3dis))
        write_fasta(output_records, output_fasta_path)

    def output_per_position_perplexity(self, input_fasta_path: str, output_tsv_path: str, batch_size: int = 4):
        """Calculates and writes per-position prediction perplexities to a TSV file."""
        aa_records = read_fasta(input_fasta_path)
        with open(output_tsv_path, 'w') as out_f:
            out_f.write("sequence_id\tposition\taa\tperplexity\n")
            with torch.no_grad():
                for i in range(0, len(aa_records), batch_size):
                    headers, raw_seqs = zip(*aa_records[i:i + batch_size])
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


def _gpu_worker(gpu_id, shard_fasta, output_fasta, checkpoint_path, progress_queue, error_event):
    """Worker for Multi-GPU sharding."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        predictor = ESM3DiPredictor(checkpoint_path, device="cuda")
        predictor.predict_from_fasta(shard_fasta, output_fasta, batch_size=2)
        progress_queue.put(("done", gpu_id))
    except Exception as e:
        progress_queue.put(("error", gpu_id, str(e)))
        error_event.set()


def run_multi_gpu_inference(input_fasta, output_3di_fasta, checkpoint_path, num_gpus):
    """Coordinates Multi-GPU inference."""
    import multiprocessing as mp
    import queue
    import tempfile
    import shutil
    from .preprocessing import _shard_fasta, _merge_fasta_outputs, _count_sequences
    from Bio import SeqIO

    if _count_sequences(input_fasta) <= 1 or num_gpus <= 1: return False

    ctx = mp.get_context('spawn')
    temp_dir = tempfile.mkdtemp(prefix="esm3di_shards_")

    try:
        shards = _shard_fasta(input_fasta, num_gpus, temp_dir)
        progress_queue, error_event = ctx.Queue(), ctx.Event()
        shard_outputs, processes = [], []

        for gpu_id, (shard_fasta, _) in enumerate(shards):
            out_path = os.path.join(temp_dir, f"shard_{gpu_id}_3di.fasta")
            shard_outputs.append((shard_fasta, out_path))
            p = ctx.Process(target=_gpu_worker,
                            args=(gpu_id, shard_fasta, out_path, checkpoint_path, progress_queue, error_event))
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
            if error_event.is_set(): raise RuntimeError("Worker failed")

        for p in processes: p.join()
        _merge_fasta_outputs(shard_outputs, output_3di_fasta, [rec.id for rec in SeqIO.parse(input_fasta, "fasta")])
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)