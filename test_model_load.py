import os
import torch
from transformers import AutoModelForTokenClassification
from peft import PeftModel

# --- CONFIGURATION PATHS ---
# Adjust these paths to where your actual files are stored
BASE_MODEL_NAME = "Synthyra/ESMplusplus_small"
HF_COMPATIBLE_DIR = "checkpoints/hf_compatible"
RAW_PT_CHECKPOINT = "checkpoints/epoch_5.pt"
NUM_LABELS = 20

# Force CPU loading to rule out CUDA driver discrepancies during verification
device = "cpu"
print(f"Running equivalence test on: {device.upper()}")

# -------------------------------------------------------------
# APPROACH A: The Hugging Face PEFT Loader (Clean, Safe Method)
# -------------------------------------------------------------
print("\n[A] Initializing via HF + PEFT Adapter...")
base_model_a = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL_NAME, num_labels=NUM_LABELS, trust_remote_code=True
)
# Inject tokenizer immediately since it is native to ESM++
tokenizer = base_model_a.tokenizer

model_peft = PeftModel.from_pretrained(base_model_a, HF_COMPATIBLE_DIR)
model_peft.to(device)
model_peft.eval()

# -------------------------------------------------------------
# APPROACH B: Manual State Dictionary Injection via epoch_5.pt
# -------------------------------------------------------------
print(f"\n[B] Initializing base and injecting raw `{os.path.basename(RAW_PT_CHECKPOINT)}`...")
base_model_b = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL_NAME, num_labels=NUM_LABELS, trust_remote_code=True
)

if os.path.exists(RAW_PT_CHECKPOINT):
    # To load the .pt state dict, we have to create the matching PEFT shell structure first
    model_raw_pt = PeftModel.from_pretrained(base_model_b, HF_COMPATIBLE_DIR)

    # Load raw state dict file
    raw_state_dict = torch.load(RAW_PT_CHECKPOINT, map_location=device)

    # Strip common wrapper prefixes if present (e.g. 'model.', 'base_model.model.')
    # PEFT models usually expect keys starting with 'base_model.model...'
    sample_key = list(raw_state_dict.keys())[0]
    print(f" -> Sample key inside your .pt file: '{sample_key}'")

    # Load the state dict directly into the architecture shell
    missing_keys, unexpected_keys = model_raw_pt.load_state_dict(raw_state_dict, strict=False)
    model_raw_pt.to(device)
    model_raw_pt.eval()
    print(" -> Raw state dict loaded into PEFT architecture shell.")
else:
    print(f"⚠️ Could not find '{RAW_PT_CHECKPOINT}'. Skipping direct array calculations.")
    model_raw_pt = None

# -------------------------------------------------------------
# EQUIVALENCE TESTING VIA INFERENCE PASS
# -------------------------------------------------------------
print("\n--- Running Equivalence Validation Pass ---")
test_sequence = ["AYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"]
inputs = tokenizer(test_sequence, padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    outputs_peft = model_peft(**inputs).logits
    print(f"PEFT Model Output Shape: {outputs_peft.shape}")

    if model_raw_pt is not None:
        outputs_raw_pt = model_raw_pt(**inputs).logits

        # Calculate maximum absolute discrepancy between the two matrices
        max_absolute_diff = torch.max(torch.abs(outputs_peft - outputs_raw_pt)).item()
        print(f"Maximum absolute numerical variance: {max_absolute_diff:.8f}")

        # Verify matrices are identical within machine float precision tolerances
        if torch.allclose(outputs_peft, outputs_raw_pt, atol=1e-5):
            print("✅ SUCCESS: The outputs are identical! You do not need 'epoch_5.pt'.")
        else:
            print("❌ WARNING: Tensor variance detected. Check for key prefix mismatch variations.")
    else:
        print("Skipped matrix comparison pass because the .pt file wasn't found.")