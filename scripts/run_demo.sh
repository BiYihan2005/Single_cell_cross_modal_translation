#!/usr/bin/env bash
set -euo pipefail

python scripts/make_synthetic_h5ad.py --output_dir data/synthetic

python src/cross_modal_translation_pipeline.py \
  --rna_train data/synthetic/adata_rna_train.h5ad \
  --met_train data/synthetic/adata_met_train.h5ad \
  --met_test data/synthetic/adata_met_test.h5ad \
  --output outputs/demo_submission.csv \
  --model_path models/demo_residual_expression_net.pt \
  --pca_components 64 \
  --hidden_dim 128 \
  --n_blocks 2 \
  --epochs 20 \
  --batch_size 32 \
  --residual_weight 0.4
