
import torch
from transformers import AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# -----------------------------
# FASTA utilities
# -----------------------------

def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Very simple FASTA parser.
    Returns list of (header_without_>, sequence_string).
    """
    records = []
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip().upper())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records

def discover_lora_target_modules(model) -> List[str]:
    """
    Automatically discover LoRA target modules by finding Linear layers in attention.
    This matches the notebook's approach of dynamically discovering modules.
    """
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ('attention' in name or 'dense' in name):
            # Extract the base module name (e.g., 'self_attn.q_proj' -> 'q_proj')
            module_name = name.split('.')[-1]
            if module_name not in target_modules:
                target_modules.append(module_name)
    
    # Fallback to common ESM attention modules if discovery fails
    if not target_modules:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'fc1', 'fc2']
    
    return target_modules


def freeze_all_but_lora_and_classifier(model):
    """
    Explicitly freeze everything except:
      - LoRA parameters (names contain 'lora_')
      - classifier head (names contain 'classifier')
    """
    for name, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_" in name or "classifier" in name:
            p.requires_grad = True



class ESM3DiModel:
    """
    Wrapper class for ESM model with LoRA adaptation for 3Di prediction.
    """
    def __init__(self, hf_model_name: str, num_labels: int, lora_r: int = 8, 
                    lora_alpha: float = 16.0, lora_dropout: float = 0.05, 
                    target_modules: List[str] = None):
        """
        Initialize ESM model with LoRA configuration.
        
        Args:
            hf_model_name: HuggingFace model identifier
            num_labels: Number of 3Di labels in vocabulary
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter  
            lora_dropout: LoRA dropout rate
            target_modules: List of module names for LoRA. If None, auto-discover.
        """
        self.hf_model_name = hf_model_name
        self.num_labels = num_labels
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Load base model
        print(f"\nLoading base model: {hf_model_name}")
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            hf_model_name,
            num_labels=num_labels,
        )
        print("✓ Base model loaded")
        
        # Determine target modules
        if target_modules:
            self.target_modules = target_modules
            print(f"\nUsing specified LoRA target modules: {target_modules}")
        else:
            print("\nAuto-discovering LoRA target modules...")
            self.target_modules = discover_lora_target_modules(self.base_model)
            print(f"Discovered target modules: {self.target_modules}")
        
        # Configure and apply LoRA
        self._setup_lora()
        self.freeze_base_model()
        print("✓ LoRA setup complete\n")
        
    def _setup_lora(self):
        """Setup LoRA configuration and wrap the base model."""
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
    
    def freeze_base_model(self):
        """Freeze all base model parameters except LoRA and classifier."""
        freeze_all_but_lora_and_classifier(self.model)

    def get_model(self):
        """Return the LoRA-wrapped model."""
        return self.model

# -----------------------------
# Dataset
# -----------------------------

class Seq3DiDataset(Dataset):
    """
    Holds (amino_acid_sequence, 3Di_label_sequence) pairs.
    Assumes 1:1 correspondence and equal lengths.

    mask_label_chars: characters in 3Di sequences that indicate "masked" positions
                      (e.g. low pLDDT). Those chars:
                      - are NOT part of label_vocab / model outputs
                      - are ignored in the loss (labels = -100).
    """
    def __init__(
        self,
        aa_fasta: str,
        three_di_fasta: str,
        mask_label_chars: str = ""
    ):
        aa_records = read_fasta(aa_fasta)
        lab_records = read_fasta(three_di_fasta)

        assert len(aa_records) == len(lab_records), "Mismatched number of sequences"

        self.items = []
        all_chars = set()
        self.mask_label_chars = set(mask_label_chars) if mask_label_chars else set()

        for (h_aa, seq_aa), (h_lab, seq_lab) in zip(aa_records, lab_records):
            if len(seq_aa) != len(seq_lab):
                raise ValueError(
                    f"Length mismatch {h_aa}/{h_lab}: {len(seq_aa)} vs {len(seq_lab)}"
                )
            self.items.append((h_aa, seq_aa, seq_lab))
            all_chars.update(seq_lab)

        # Build vocab from all non-masked characters
        label_chars = sorted(ch for ch in all_chars if ch not in self.mask_label_chars)
        self.label_vocab = label_chars
        self.char2idx = {c: i for i, c in enumerate(self.label_vocab)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # (header, aa_seq, 3di_seq)
        return self.items[idx]


# -----------------------------
# Collate with HF tokenizer
# -----------------------------

def make_collate_fn(tokenizer, char2idx, mask_label_chars: str = ""):
    """
    - Tokenizes AA sequences with HF ESM tokenizer.
    - Uses special_tokens_mask to align per-residue 3Di labels
      to non-special tokens.
    - Positions whose 3Di label is in mask_label_chars are set to -100
      (ignored in the loss) and do NOT belong to label_vocab.
    """
    mask_set = set(mask_label_chars) if mask_label_chars else set()

    def collate(batch):
        headers, aa_seqs, label_seqs = zip(*batch)

        enc = tokenizer(
            list(aa_seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        input_ids = enc["input_ids"]              # [B, T]
        attention_mask = enc["attention_mask"]    # [B, T]
        special_mask = enc["special_tokens_mask"] # [B, T]

        batch_size, max_len = input_ids.shape
        labels = torch.full(
            (batch_size, max_len),
            -100,  # ignore_index for CE
            dtype=torch.long,
        )

        for i, lab_seq in enumerate(label_seqs):
            k = 0  # index into label sequence
            for j in range(max_len):
                if special_mask[i, j] == 1:
                    # CLS/EOS/PAD etc.
                    labels[i, j] = -100
                else:
                    if k < len(lab_seq):
                        ch = lab_seq[k]
                        if ch in mask_set:
                            # masked (e.g. low pLDDT) -> ignore in loss
                            labels[i, j] = -100
                        else:
                            try:
                                labels[i, j] = char2idx[ch]
                            except KeyError:
                                raise ValueError(
                                    f"Label char '{ch}' not in vocabulary and not in "
                                    f"mask_label_chars; check your data."
                                )
                        k += 1
                    else:
                        labels[i, j] = -100
            if k != len(lab_seq):
                # Safety check: we should have consumed all labels
                raise ValueError(
                    f"Did not consume all labels for sequence {headers[i]}: "
                    f"used {k}, have {len(lab_seq)}"
                )

        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch_out

    return collate

