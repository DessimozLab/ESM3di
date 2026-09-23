"""
Runner implementation for ESM3Di using the esm3di Python package.
"""

from pathlib import Path
import tempfile
from typing import List, Optional
import logging

from models.base import BaseRunner

try:
    from esm3di.inference import ESM3DiPredictor
except ImportError:
    ESM3DiPredictor = None


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


class ESM3DiRunner(BaseRunner):
    """Benchmark runner for ESM3Di model."""

    def __init__(
        self,
        model_name: str = "esm3di",
        device: str = "cuda",
        batch_size: int = 8,
    ):
        super().__init__(model_name=model_name, device=device)
        self.batch_size = batch_size
        self.predictor: Optional[ESM3DiPredictor] = None

    def load_model(self) -> None:
        """Loads ESM3Di pretrained model into memory."""
        if ESM3DiPredictor is None:
            raise ImportError(
                "esm3di package is not installed. Ensure it is installed in your active environment."
            )

        #print(f"[+] Loading {self.model_name} model...")
        self.predictor = ESM3DiPredictor.from_pretrained()
        self.is_loaded = True

    def _run_inference(self, fasta_path: Path) -> None:
            """
            Runs ESM3Di inference on the input FASTA file using native batching.
            Pure timing of model inference and output generation.

            Args:
                fasta_path (Path): Path to the input FASTA file.
            """
            with tempfile.NamedTemporaryFile(suffix=".fasta") as tmp_out:
                # Leverage ESM3Di's native predict_fasta pipeline
                self.predictor.predict_fasta(
                    str(fasta_path),
                    tmp_out.name,
                    batch_size=self.batch_size,
                )