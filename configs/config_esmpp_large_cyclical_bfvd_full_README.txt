================================================================================
CONFIG: config_esmpp_large_cyclical_bfvd_full.json
================================================================================

DESCRIPTION:
  ESMplusplus Large model with pLDDT-weighted Cyclical Focal Loss
  Based on: config_esmpp_large_lion_bfvd_full.json
  Dataset: BFVD full (350K+ sequences)

LOSS FUNCTION:
  PLDDTWeightedCyclicalFocalLoss
  - Combines cyclical focal loss dynamics with pLDDT confidence weighting
  - gamma_pos: 0.0 (no suppression of correct predictions)
  - gamma_neg: 4.0 (strong suppression of easy negatives)
  - gamma_hc: 0.0 (no hard-class boost)
  - cyclical_factor: 2.0 (full cyclical schedule)
  - label_smoothing: 0.1 (mild regularization)
  - plddt_min_bin: 5 (ignore pLDDT < 50)
  - plddt_weight_exponent: 1.5 (sharpen confidence contrast)

TRAINING SCHEDULE (25 epochs):
  Epoch 0-8:   Focus on broad learning (eta → 1.0 to 0.3)
  Epoch 9-16:  Focus on hard examples (eta → 0.1)
  Epoch 17-25: Cyclical refinement (eta → 0.0 to 1.1)

MODEL SETUP:
  - Base: Synthyra/ESMplusplus_large
  - LoRA: r=128, alpha=256, dropout=0.05
  - No CNN head (linear classifier)

OPTIMIZER:
  - Lion optimizer
  - Learning rate: 3e-5
  - Weight decay: 0.1
  - Betas: (0.9, 0.99)
  - Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
  - Warmup: 5% of training

BATCH CONFIGURATION:
  - Batch size: 6 per GPU
  - Gradient accumulation: 9 steps
  - Effective batch size: 54
  - Custom epoch: 50,000 samples per epoch

HARDWARE:
  - Multi-GPU training enabled
  - Mixed precision (FP16)
  - CUDA device: cuda:0

OUTPUT:
  - Checkpoints: checkpoints_esmpp_large_cyclical_bfvd_full/

================================================================================
USAGE
================================================================================

Start training:
  python -m esm3di.esmretrain --config config_esmpp_large_cyclical_bfvd_full.json

Resume from checkpoint:
  python -m esm3di.esmretrain \
    --config config_esmpp_large_cyclical_bfvd_full.json \
    --resume checkpoints_esmpp_large_cyclical_bfvd_full/epoch_X.pt

Monitor training:
  tensorboard --logdir checkpoints_esmpp_large_cyclical_bfvd_full/

================================================================================
EXPECTED BEHAVIOR
================================================================================

Early epochs (0-8):
  - Loss starts high as model learns all positions
  - Accuracy climbs quickly
  - Hard-class weight dominates (broad learning)

Mid epochs (9-16):
  - Loss continues to decrease
  - Model refines difficult predictions
  - Asymmetric weight dominates (mistake correction)

Late epochs (17-25):
  - Fine-tuning with cyclical schedule
  - Model balances learning vs refinement
  - Accuracy should plateau at high values

pLDDT-stratified accuracy:
  - Track improvement at different confidence levels
  - High confidence (pLDDT ≥80) should reach ~85-90%
  - Medium confidence (pLDDT ≥60) should reach ~75-85%
  - Low confidence regions have less impact due to weighting

================================================================================
COMPARISON TO ORIGINAL LION CONFIG
================================================================================

Original (config_esmpp_large_lion_bfvd_full.json):
  - Regular Focal Loss (gamma=3.0)
  - Gamma scheduler (increases gamma on plateau)
  - Static loss weighting strategy

This config (cyclical):
  - Cyclical Focal Loss with dynamic weighting
  - No gamma scheduler (not compatible)
  - Epoch-dependent learning strategy
  - Expected benefits:
    * Better handling of class imbalance
    * More stable training
    * Potential for higher final accuracy

================================================================================
