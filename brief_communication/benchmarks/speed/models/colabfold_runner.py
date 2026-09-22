"""
Benchmark runner for the full AlphaFold2 -> 3Di structure prediction pipeline using
LocalColabFold (colabfold_batch) and Foldseek structure-to-3Di conversion.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

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
        """
        Args:
            model_name (str): Identifier name for logging.
            device (str): Execution target device.
            num_recycles (int): AlphaFold2 recycling iterations (default: 3).
            num_models (int): Number of AF2 structural models per sequence (default: 1).
            msa_mode (str): MSA search mode ('single_sequence' for single-sequence prediction,
                            or 'mmseqs2_uniref_env' for full web-server MSA querying).
            use_amber (bool): Whether to perform CPU AMBER forcefield relaxation (default: False).
        """
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

    def _run_inference(self, fasta_path: Path) -> None:
        """
        Runs colabfold_batch to generate PDB structures, then converts PDBs
        to 3Di sequence tokens using Foldseek createdb.

        Args:
            fasta_path (Path): Path to input sequence FASTA.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_dir = Path(tmp_dir) / "colabfold_pdbs"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            tmp_db = Path(tmp_dir) / "foldseek_structure_db"

            # 1. Build colabfold_batch command
            cmd_cf = [
                "colabfold_batch",
                str(fasta_path),
                str(pdb_dir),
                "--num-recycle", str(self.num_recycles),
                "--num-models", str(self.num_models),
                "--msa-mode", self.msa_mode,
            ]

            if self.use_amber:
                cmd_cf.append("--amber")

            # Prevent greedy JAX GPU VRAM pre-allocation
            env = os.environ.copy()
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

            # 2. Run ColabFold structure prediction
            res_cf = subprocess.run(
                cmd_cf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
            )
            if res_cf.returncode != 0:
                raise RuntimeError(f"ColabFold execution failed: {res_cf.stderr}")

            # 3. Convert predicted PDB structures into Foldseek 3Di database
            cmd_fs = [
                "foldseek", "createdb",
                str(pdb_dir),
                str(tmp_db),
                "-v", "0",
            ]
            res_fs = subprocess.run(
                cmd_fs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
            )
            if res_fs.returncode != 0:
                raise RuntimeError(f"Foldseek createdb failed on predicted structures: {res_fs.stderr}")