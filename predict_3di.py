#!/usr/bin/env python3
"""
Example script for predicting 3Di sequences from amino acid FASTA files.
"""

from ESM3di_model import ESM3DiModel


def main():
    # Initialize model with same configuration as training
    model = ESM3DiModel(
        hf_model_name="facebook/esm2_t33_650M_UR50D",
        num_labels=20,  # 20 standard 3Di characters
        use_cnn_head=False,  # Set to True if trained with CNN head
    )
    
    # Predict using trained checkpoint
    model.predict_from_fasta(
        input_fasta_path="test_data_aa.fasta",
        output_fasta_path="predicted_3di.fasta",
        model_checkpoint_path="checkpoints/epoch_3.pt",
        batch_size=4,
        device=None  # Auto-detect (cuda/cpu)
    )


if __name__ == "__main__":
    main()

