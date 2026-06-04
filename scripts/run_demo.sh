#!/usr/bin/env bash
# 运行合成数据 demo。
# 使用方式：
#   bash scripts/run_demo.sh

set -e

echo "Step 1/2: 生成合成 h5ad 数据"
python3 scripts/make_synthetic_h5ad.py --output_dir data/example

echo "Step 2/2: 运行跨模态预测 pipeline"
python3 src/cross_modal_translation_pipeline.py \
  --rna_train data/example/adata_rna_train.h5ad \
  --met_train data/example/adata_met_train.h5ad \
  --met_test data/example/adata_met_test.h5ad \
  --output outputs/demo_submission.csv \
  --model_path models/demo_residual_net.pt \
  --pca_components 64 \
  --hidden_dim 128 \
  --n_blocks 2 \
  --epochs 10 \
  --batch_size 32 \
  --predict_batch_size 64 \
  --residual_weight 0.4 \
  --seed 42

echo "Demo 运行完成：outputs/demo_submission.csv"
