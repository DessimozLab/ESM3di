#!/usr/bin/env python3
"""
colabfold_runner.py

Benchmark runner for the full AlphaFold2 -> 3Di structure prediction pipeline using
LocalColabFold (colabfold_batch) and Foldseek structure-to-3Di conversion.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from tqdm import tqdm

from models.base import BaseRunner


class ColabFoldRunner(BaseRunner):
    """
    Benchmark runner for ColabFold (AlphaFold2) structure prediction
    followed by Foldseek 3Di token extraction.
    """

    def __init__(
        self,
        model_name: str = "colabfold",
        device: str = "cuda",
        num_recycles: int = 3,
        num_models: int = 1,
        msa_mode: str = "single_sequence",
        use_amber: bool = False,
    ):
        super().__init__(model_name=model_name, device=device)
        self.num_recycles = num_recycles
        self.num_models = num_models
        self.msa_mode = msa_mode
        self.use_amber = use_amber

    def load_model(self) -> None:
        """
        Verifies that 'colabfold_batch' and 'foldseek' binaries exist in PATH.
        """
        if not shutil.which("colabfold_batch"):
            raise RuntimeError(
                "Executable 'colabfold_batch' not found in PATH. Ensure LocalColabFold is installed."
            )
        if not shutil.which("foldseek"):
            raise RuntimeError(
                "Executable 'foldseek' not found in PATH. Ensure Foldseek is installed."
            )

        print(f"[+] Verified binaries for {self.model_name}: 'colabfold_batch' and 'foldseek'.")
        self.is_loaded = True

    def _count_fasta_sequences(self, fasta_path: Path) -> int:
        """Counts the total number of sequences in the input FASTA file."""
        count = 0
        with open(fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(">"):
                    count += 1
        return max(1, count)

    def _run_inference(self, fasta_path: Path) -> None:
        """
        Runs colabfold_batch to generate PDB structures. Uses a background thread
        for live progress tracking to ensure zero timing latency on main thread execution.
        """
        import threading
        import time

        total_seqs = self._count_fasta_sequences(fasta_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_dir = Path(tmp_dir) / "colabfold_pdbs"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            tmp_db = Path(tmp_dir) / "foldseek_structure_db"

            cmd_cf = [
                "colabfold_batch",
                str(fasta_path),
                str(pdb_dir),
                "--num-recycle", str(self.num_recycles),
                "--num-models", str(self.num_models),
                "--msa-mode", self.msa_mode,
                "--sort-queries-by", "length",
            ]

            if self.use_amber:
                cmd_cf.append("--amber")

            env = os.environ.copy()
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["PYTHONUNBUFFERED"] = "1"

            # Launch process
            process = subprocess.Popen(
                cmd_cf,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            # Define background thread worker for non-blocking progress updates
            stop_monitoring = threading.Event()

            def _track_progress():
                completed = 0
                with tqdm(total=total_seqs, desc="ColabFold Progress", unit="seq") as pbar:
                    while not stop_monitoring.is_set():
                        rank1_files = list(pdb_dir.glob("*_rank_001_*.pdb")) + \
                                     list(pdb_dir.glob("*_rank_001_*.cif")) + \
                                     list(pdb_dir.glob("*_rank_1_*.pdb"))
                        current = len(rank1_files)
                        if current > completed:
                            pbar.update(current - completed)
                            completed = current
                        time.sleep(1.0)  # Polling interval in separate thread

                    # Final update catch-up
                    rank1_files = list(pdb_dir.glob("*_rank_001_*.pdb")) + \
                                 list(pdb_dir.glob("*_rank_001_*.cif")) + \
                                 list(pdb_dir.glob("*_rank_1_*.pdb"))
                    current = len(rank1_files)
                    if current > completed:
                        pbar.update(current - completed)

            # Start background progress monitoring
            monitor_thread = threading.Thread(target=_track_progress, daemon=True)
            monitor_thread.start()

            try:
                # Main thread blocks cleanly — EXACT timing, zero sleep delay
                stdout, _ = process.communicate()
            finally:
                stop_monitoring.set()
                monitor_thread.join()

            if process.returncode != 0:
                error_msg = stdout[-2000:] if stdout else "Unknown error"
                raise RuntimeError(f"ColabFold execution failed:\n{error_msg}")

            # Convert predicted PDB structures into Foldseek 3Di database
            cmd_fs = ["foldseek", "createdb", str(pdb_dir), str(tmp_db), "-v", "0"]
            res_fs = subprocess.run(cmd_fs, capture_output=True, text=True, env=env)
            if res_fs.returncode != 0:
                raise RuntimeError(f"Foldseek createdb failed: {res_fs.stderr}")