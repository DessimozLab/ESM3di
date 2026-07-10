import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from typing import List, Tuple, Union, Optional
from peft import get_peft_model, LoraConfig, TaskType
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

        # 1. Resolve local path or handle remote streaming downloads
        if model_checkpoint_path == "default":
            print(f"Using default model. Checking remote cache at {DEFAULT_HF_REPO}...")
            resolved_path = hf_hub_download(repo_id=DEFAULT_HF_REPO, filename=DEFAULT_FILENAME)
        elif os.path.exists(model_checkpoint_path):
            resolved_path = model_checkpoint_path
        else:
            print(f"Downloading from HF Hub: {model_checkpoint_path}...")
            resolved_path = hf_hub_download(repo_id=model_checkpoint_path, filename=DEFAULT_FILENAME)

        # 2. Safely unpack checkpoint state and extract configuration metadata
        checkpoint_dict = torch.load(resolved_path, map_location="cpu")
        is_dict = isinstance(checkpoint_dict, dict)

        args = checkpoint_dict.get('args', {}) if is_dict else {}
        label_vocab = checkpoint_dict.get('label_vocab', list("ACDEFGHIKLMNPQRSTVWY")) if is_dict else list(
            "ACDEFGHIKLMNPQRSTVWY")
        hf_model_name = args.get('hf_model_name', args.get('hf_model', 'facebook/esm2_t12_35M_UR50D'))

        # 3. Reconstruct exact model configuration arguments
        # 3. Reconstruct exact model configuration arguments
        model_kwargs = {
            "hf_model_name": hf_model_name,
            "num_labels": len(label_vocab),
            "lora_r": args.get("lora_r", 8),
            "lora_alpha": args.get("lora_alpha", 16.0),

            # CNN Configs
            "cnn_num_layers": args.get("cnn_num_layers", 2),
            "cnn_kernel_size": args.get("cnn_kernel_size", 3),
            "cnn_dropout": args.get("cnn_dropout", 0.1),

            # Transformer Configs
            "transformer_head_dim": args.get("transformer_head_dim", 256),
            "transformer_head_layers": args.get("transformer_head_layers", 2),
            "transformer_head_dropout": args.get("transformer_head_dropout", 0.1),
            "transformer_head_num_heads": args.get("transformer_head_num_heads", None),

            # Iterative Configs
            "iterative_head_max_iterations": args.get("iterative_head_max_iterations", 5),
            "iterative_head_halt_threshold": args.get("iterative_head_halt_threshold", 0.95),
            "use_positional_encoding": args.get("use_positional_encoding", True),
            "use_hidden_state_feedback": args.get("use_hidden_state_feedback", True),
            "use_gru_gate": args.get("use_gru_gate", False),

            # PLDDT Configs
            "plddt_prediction_mode": args.get("plddt_prediction_mode", "classification"),
            "plddt_num_bins": args.get("plddt_num_bins", 10),

            # Feature Flags
            **{k: args.get(k, False) for k in [
                "use_cnn_head", "use_transformer_head",
                "use_iterative_transformer_head", "use_plddt_prediction_head"
            ]}
        }

        # 4. Build clean custom model and wrap it natively in PEFT
        print(f"Instantiating native structural architecture for backbone: {hf_model_name}")
        raw_custom_model = ESM3DiModel(**model_kwargs)

        lora_config = LoraConfig(
            task_type=None,  # Custom architecture task routing
            r=model_kwargs["lora_r"],
            lora_alpha=model_kwargs["lora_alpha"],
            lora_dropout=args.get("lora_dropout", 0.05),
            target_modules=["query", "key", "value", "dense"],
            modules_to_save=["classifier", "plddt_head"]  # Natively tracks and loads the custom heads
        )
        self.model = get_peft_model(raw_custom_model, lora_config)

        # 5. Load and clean weights (handling DataParallel and legacy PEFT naming shifts)
        raw_state_dict = checkpoint_dict.get('model_state_dict', checkpoint_dict) if is_dict else checkpoint_dict

        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            k_clean = k.replace("module.", "", 1)

            # Map legacy classifier naming straight to modern PEFT layout
            if k_clean == "base_model.model.classifier.original_weight":
                k_clean = "base_model.model.classifier.original_module.weight"
            elif k_clean == "base_model.model.classifier.original_bias":
                k_clean = "base_model.model.classifier.original_module.bias"

            clean_state_dict[k_clean] = v

        # 6. State Validation Pass
        print("Loading weights into clean structural layout...")
        info = self.model.load_state_dict(clean_state_dict, strict=False)

        # Verify that ONLY the standard, un-utilized ESM-2 rotary embedding weight is missing
        expected_missing = {"base_model.model.esm.embeddings.position_embeddings.weight"}
        actual_missing = set(info.missing_keys) - expected_missing

        if actual_missing or info.unexpected_keys:
            raise RuntimeError(
                f"Critical structural mismatch caught during weight loading!\n"
                f"Missing Keys: {actual_missing}\n"
                f"Unexpected Keys: {info.unexpected_keys}"
            )

        print("✓ Production Release Verification Passed: 100% true weight match achieved.")

        # 7. Finalize runtime execution engine
        self.model = self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        self.idx2char = {i: c for i, c in enumerate(label_vocab)}

        print("✓ Production Release Verification Passed: 100% native weight match achieved.")


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