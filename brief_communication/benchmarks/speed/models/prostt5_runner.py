"""
Benchmark runner for ProstT5 using Foldseek's native C++ / GGUF implementation.
Executes high-throughput quantized 3Di generation via Foldseek CLI.

# 1. Make sure Foldseek is installed and available in PATH:
conda install -c conda-forge -c bioconda foldseek
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import List

from models.base import BaseRunner


class ProstT5Runner(BaseRunner):
    """Benchmark runner for Foldseek's native GGUF ProstT5 implementation."""

    def __init__(
        self,
        model_name: str = "prostt5",
        weights_dir: Path = Path("weights/prostt5_model"),
        device: str = "cuda",
        threads: int = 8,
    ):
        super().__init__(model_name=model_name, device=device)
        self.weights_dir = weights_dir
        self.threads = threads

    def load_model(self) -> None:
        """
        Verifies Foldseek binary installation and ensures ProstT5 GGUF weights
        are downloaded and cached locally.
        """
        # 1. Verify foldseek executable exists
        if not shutil.which("foldseek"):
            raise RuntimeError(
                "Foldseek binary not found in PATH. Install via conda: "
                "'conda install -c conda-forge -c bioconda foldseek'"
            )

        # 2. Check and download ProstT5 GGUF weights if not cached
        self.weights_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Foldseek creates a folder with binary database files
        if not (self.weights_dir.exists() and any(self.weights_dir.iterdir())):
            print(f"[+] Downloading Foldseek ProstT5 GGUF weights to {self.weights_dir}...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                cmd = [
                    "foldseek", "databases", "ProstT5",
                    str(self.weights_dir),
                    tmp_dir
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"Failed to download ProstT5 model via Foldseek: {res.stderr}")
            print("[+] ProstT5 GGUF model download complete.")
        else:
            print(f"[+] Using cached Foldseek ProstT5 weights at: {self.weights_dir}")

        self.is_loaded = True

    def _run_inference(self, fasta_path: Path) -> List[str]:
        """
        Executes native C++ Foldseek createdb with --prostt5-model.

        Args:
            fasta_path (Path): Path to input sequence FASTA.

        Returns:
            List[str]: List of predicted 3Di string sequences extracted from DB.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_db = Path(tmp_dir) / "out_3di_db"

            # 1. Execute Foldseek 3Di generation
            cmd = [
                "foldseek", "createdb",
                str(fasta_path),
                str(tmp_db),
                "--prostt5-model", str(self.weights_dir),
                "--threads", str(self.threads),
                "-v", "0",  # Suppress verbose log output
            ]

            env = os.environ.copy()
            # Ensure GPU usage if available
            if self.device == "cuda":
                cmd.extend(["--gpu", "1"])

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if res.returncode != 0:
                raise RuntimeError(f"Foldseek ProstT5 execution failed: {res.stderr}")