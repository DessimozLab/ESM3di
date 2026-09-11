import logging
from esm3di.inference import ESM3DiPredictor
from esm3di.io import fasta2foldseek
from pathlib import Path

# Enable logging output for standard Python scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Initialize predictor (optionally pass revision="46c5f7d")
predictor = ESM3DiPredictor.from_pretrained(revision="46c5f7d")

# 1. In-Memory Sequence Prediction
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
p_3di = predictor.predict(sequence)
print(f"3Di Output: {p_3di}\n")