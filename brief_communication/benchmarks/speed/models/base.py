"""
Abstract Base Class for model benchmark runners. Provides uniform timing,
CUDA synchronization, VRAM tracking, and metric collection across all models.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class BenchmarkResult:
    """Dataclass holding standardized output metrics for benchmarking."""

    model_name: str
    dataset_name: str
    num_sequences: int
    total_residues: int
    wall_time_seconds: float
    sequences_per_second: float
    residues_per_second: float
    peak_vram_gb: float
    device: str

    def to_dict(self) -> Dict:
        """Converts result metrics to a dictionary for CSV exporting."""
        return asdict(self)


class BaseRunner(ABC):
    """Abstract Base Class for all model runners."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Abstract method to load model weights onto GPU/CPU and run warm-up.
        Must set self.is_loaded = True at the end.
        """
        pass

    @abstractmethod
    def _run_inference(self, fasta_path: Path) -> List[str]:
        """
        Abstract method containing model-specific execution loop.
        Should process the FASTA file and return predicted 3Di token sequences.

        Args:
            fasta_path (Path): Path to the FASTA file.

        Returns:
            List[str]: List of predicted 3Di string sequences.
        """
        pass

    def run_benchmark(self, fasta_path: Path) -> BenchmarkResult:
        if not self.is_loaded:
            self.load_model()

        # 1. Parse input FASTA metrics
        num_seqs, num_res = self._get_fasta_stats(fasta_path)

        # 2. Reset VRAM tracking if CUDA is available
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # 3. Measure inference time
        start_time = time.perf_counter()
        self._run_inference(fasta_path)
        wall_time = time.perf_counter() - start_time

        # 4. Measure peak VRAM
        peak_vram_gb = 0.0
        if torch.cuda.is_available() and "cuda" in self.device:
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # 5. Calculate throughput metrics
        seq_per_sec = num_seqs / wall_time if wall_time > 0 else 0.0
        res_per_sec = num_res / wall_time if wall_time > 0 else 0.0

        return BenchmarkResult(
            model_name=self.model_name,
            dataset_name=fasta_path.name,
            num_sequences=num_seqs,
            total_residues=num_res,
            wall_time_seconds=wall_time,
            sequences_per_second=seq_per_sec,
            residues_per_second=res_per_sec,
            peak_vram_gb=peak_vram_gb,
            device=self.device
        )

    def _get_fasta_stats(self, fasta_path: Path) -> Tuple[int, int]:
        """Reads FASTA file to compute sequence count and total residue count."""
        num_sequences = 0
        total_residues = 0

        with open(fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    num_sequences += 1
                else:
                    total_residues += len(line)

        return num_sequences, total_residues

    def warm_up(self, dummy_length: int = 300) -> None:
        """
        Helper warm-up method to execute 1-2 dummy passes on the GPU
        to trigger CUDA context initialization before timing starts.
        """
        print(f"[+] Running CUDA warm-up for {self.model_name}...")
        dummy_seq = "A" * dummy_length
        
        # Create temporary dummy FASTA
        tmp_path = Path("/tmp/warmup_dummy.fasta")
        with open(tmp_path, "w") as f:
            f.write(f">warmup\n{dummy_seq}\n")

        # Execute un-timed forward pass
        try:
            self._run_inference(tmp_path)
            if self.device == "cuda":
                torch.cuda.synchronize()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"[+] Warm-up completed for {self.model_name}.")