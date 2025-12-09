
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Try to import ESMC for EvolutionaryScale ESMC models
try:
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    ESMC_AVAILABLE = True
except ImportError:
    ESMC_AVAILABLE = False


class ESMCModelWrapper(nn.Module):
    """
    Wrapper for ESMC models to provide TokenClassification interface.
    ESMC models use a different architecture from ESM-2.
    """
    def __init__(self, esmc_model, num_labels):
        super().__init__()
        self.esmc = esmc_model
        self.num_labels = num_labels

        # Add classification head on top of ESMC embeddings
        hidden_size = esmc_model.transformer.d_model
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Create a minimal config for compatibility
        class Config:
            def __init__(self, hidden_size, num_labels):
                self.hidden_size = hidden_size
                self.num_labels = num_labels

        self.config = Config(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # ESMC forward expects sequence_tokens and sequence_id
        # sequence_id is a boolean mask (True = valid token)
        sequence_id = attention_mask.bool() if attention_mask is not None else None

        # Get ESMC embeddings
        esmc_output = self.esmc(
            sequence_tokens=input_ids,
            sequence_id=sequence_id
        )

        # esmc_output.embeddings has shape (batch, seq_len, hidden_size)
        sequence_output = esmc_output.embeddings

        # Apply classification head
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


# -----------------------------
# CNN Classification Head
# -----------------------------

class CNNClassificationHead(nn.Module):
    """
    Multi-layer CNN classification head for per-token classification.
    Applies 1D convolutions over the sequence dimension.
    """
    def __init__(self, hidden_size: int, num_labels: int, 
                 num_layers: int = 2, kernel_size: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            hidden_size: Input dimension from encoder
            num_labels: Number of output classes
            num_layers: Number of CNN layers (default: 2)
            kernel_size: Convolution kernel size (default: 3)
            dropout: Dropout rate between layers (default: 0.1)
        """
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        for i in range(num_layers):
            in_channels = hidden_size if i == 0 else hidden_size
            out_channels = hidden_size
            
            # Conv1d expects (batch, channels, length)
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            ))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.cnn_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, seq_len, num_labels)
        """
        # Transpose for Conv1d: (batch, hidden, seq_len)
        x = hidden_states.transpose(1, 2)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Transpose back: (batch, seq_len, hidden)
        x = x.transpose(1, 2)
        
        # Final classification
        logits = self.classifier(x)
        return logits


class ESMWithCNNHead(nn.Module):
    """
    Wrapper that replaces the standard classifier with a CNN head.
    """
    def __init__(self, base_model, cnn_head):
        super().__init__()
        self.base_model = base_model
        self.cnn_head = cnn_head
        
        # Store config for compatibility
        self.config = base_model.config
        
    def forward(self, input_ids, attention_mask=None, labels=None, 
                **kwargs):
        # Get encoder outputs (without classification head)
        outputs = self.base_model.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        sequence_output = outputs[0]  # (batch, seq_len, hidden_size)
        
        # Apply CNN classification head
        logits = self.cnn_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.num_labels),
                labels.view(-1)
            )
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)


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
      - CNN head (names contain 'cnn_head')
    """
    for name, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_" in name or "classifier" in name or "cnn_head" in name:
            p.requires_grad = True



class ESM3DiModel:
    """
    Wrapper class for ESM model with LoRA adaptation for 3Di prediction.
    Supports both ESM-2 (HuggingFace) and ESMC (EvolutionaryScale) models.
    """
    def __init__(self, hf_model_name: str, num_labels: int, lora_r: int = 8,
                    lora_alpha: float = 16.0, lora_dropout: float = 0.05,
                    target_modules: List[str] = None,
                    use_cnn_head: bool = False, cnn_num_layers: int = 2,
                    cnn_kernel_size: int = 3, cnn_dropout: float = 0.1):
        """
        Initialize ESM model with LoRA configuration.

        Args:
            hf_model_name: HuggingFace model identifier or ESMC model name
            num_labels: Number of 3Di labels in vocabulary
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names for LoRA. If None, auto-discover.
            use_cnn_head: Whether to use CNN classification head instead of linear
            cnn_num_layers: Number of CNN layers (if use_cnn_head=True)
            cnn_kernel_size: CNN kernel size (if use_cnn_head=True)
            cnn_dropout: Dropout rate for CNN layers (if use_cnn_head=True)
        """
        self.hf_model_name = hf_model_name
        self.num_labels = num_labels
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_cnn_head = use_cnn_head
        self.cnn_num_layers = cnn_num_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_dropout = cnn_dropout

        # Detect model type and load accordingly
        self.is_esmc = self._is_esmc_model(hf_model_name)

        if self.is_esmc:
            self._load_esmc_model()
        else:
            self._load_esm2_model()

        # Determine target modules
        if self.is_esmc:
            # ESMC models have different architecture
            if target_modules:
                self.target_modules = target_modules
            else:
                # Default ESMC attention modules
                self.target_modules = ['attn.q_proj', 'attn.k_proj',
                                       'attn.v_proj', 'attn.out_proj']
            print(f"\nUsing LoRA target modules for ESMC: "
                  f"{self.target_modules}")
        else:
            # ESM-2 models
            if target_modules:
                self.target_modules = target_modules
                print(f"\nUsing specified LoRA target modules: "
                      f"{target_modules}")
            else:
                print("\nAuto-discovering LoRA target modules...")
                self.target_modules = discover_lora_target_modules(
                    self.base_model)
                print(f"Discovered target modules: {self.target_modules}")

        # Configure and apply LoRA
        self._setup_lora()

        # Optionally add CNN classification head
        if self.use_cnn_head:
            self._setup_cnn_head()
            print(f"✓ CNN classification head added "
                  f"({cnn_num_layers} layers)")

        self.freeze_base_model()
        print("✓ LoRA setup complete\n")

    def _is_esmc_model(self, model_name: str) -> bool:
        """
        Check if the model name refers to an ESMC model.
        """
        esmc_identifiers = ['esmc-300m', 'esmc-600m', 'esmc_300m',
                            'esmc_600m', 'esmc-6b', 'esmc_6b']
        return any(identifier in model_name.lower()
                   for identifier in esmc_identifiers)

    def _load_esmc_model(self):
        """
        Load ESMC model from EvolutionaryScale.
        """
        if not ESMC_AVAILABLE:
            raise ImportError(
                "ESMC models require the 'esm' library from "
                "EvolutionaryScale. Install it with: pip install esm"
            )

        print(f"\nLoading ESMC model: {self.hf_model_name}")

        # Map common names to ESMC model names
        model_name_map = {
            'esmc-300m-2024-12': 'esmc_300m',
            'esmc-600m-2024-12': 'esmc_600m',
            'esmc-300m': 'esmc_300m',
            'esmc-600m': 'esmc_600m',
        }

        esmc_model_name = model_name_map.get(self.hf_model_name, 'esmc_300m')

        # Load ESMC base model
        esmc_base = ESMC.from_pretrained(esmc_model_name)
        print("✓ ESMC base model loaded")

        # Wrap in our interface
        self.base_model = ESMCModelWrapper(esmc_base, self.num_labels)
        print("✓ ESMC wrapper initialized")

    def _load_esm2_model(self):
        """
        Load ESM-2 model from HuggingFace.
        """
        # Load base model
        print(f"\nLoading ESM-2 model: {self.hf_model_name}")
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            self.hf_model_name,
            num_labels=self.num_labels,
        )
        print("✓ Base model loaded")
        
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
    
    def _setup_cnn_head(self):
        """Replace standard classifier with CNN classification head."""
        # Get hidden size from base model config
        hidden_size = self.base_model.config.hidden_size
        
        # Create CNN head
        cnn_head = CNNClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            num_layers=self.cnn_num_layers,
            kernel_size=self.cnn_kernel_size,
            dropout=self.cnn_dropout
        )
        
        # Wrap model with CNN head
        self.model = ESMWithCNNHead(self.model, cnn_head)
    
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

