# Single-Cell Cross-Modal Translation

Interpretable prototype-residual pipeline for predicting single-cell RNA expression from DNA methylation features.

This repository implements a compact research-style workflow for **single-cell cross-modal translation**. The core task is to infer an RNA expression matrix when only the DNA methylation modality is observed. Instead of directly fitting a black-box neural network from methylation features to thousands of RNA genes, the pipeline decomposes the prediction into a biologically interpretable **cell-type prototype component** and a neural **residual correction component**.

## Motivation

Single-cell multi-omics data provide complementary views of cellular state. DNA methylation reflects regulatory and epigenetic structure, while RNA expression captures the downstream transcriptional state. In practice, one modality may be missing, expensive to measure, or available only for a subset of cells. Cross-modal translation asks whether the observed modality can be used to reconstruct the missing one.

This project focuses on the mapping:

$$
X^{\mathrm{met}}_i \longrightarrow Y^{\mathrm{rna}}_i,
$$

where $$X^{\mathrm{met}}_i$$ denotes methylation features for cell $$i$$ and $$Y^{\mathrm{rna}}_i$$ denotes its RNA expression vector.

## Method Overview

The pipeline follows a prototype-residual design:

1. **Methylation representation**: impute missing values, remove zero-variance features, standardize features, and project methylation data into a low-dimensional PCA space.
2. **Soft cell-type inference**: train a methylation-based classifier to estimate each cell's probability over major cell types.
3. **RNA prototype reconstruction**: compute the mean RNA expression profile for each major cell type and use soft probabilities to form a first-stage RNA prediction.
4. **Residual neural correction**: train a residual MLP to learn the remaining cell-level variation not explained by cell-type prototypes.
5. **Non-negative post-processing**: clip predicted expression values to satisfy the non-negativity of RNA expression measurements.

The final prediction is:

$$
\begin{aligned}
\widehat{Y}_{\mathrm{test}}
&=
P_{\mathrm{test}} C
+
\alpha g_{\theta}\!\left(Z_{\mathrm{test}}\right).
\end{aligned}
$$

where $P_{\mathrm{test}}$ is the soft cell-type probability matrix, $C$ is the RNA prototype matrix, $Z_{\mathrm{test}}$ is the PCA representation of methylation features, $g_{\theta}$ is the residual neural network, and $\alpha$ controls the residual correction strength.

## Why This Design Is Useful

A direct neural regression model must learn both coarse cell-type identity and fine-grained cell-level deviations in one step. This is difficult because RNA output is high-dimensional and single-cell data are noisy. The prototype-residual design separates these two sources of signal:

- the **prototype term** captures cell-type-level transcriptional structure;
- the **residual term** captures intra-type heterogeneity, transitional states, and methylation-associated deviations.

This makes the model more interpretable than a pure end-to-end regressor while still allowing nonlinear correction through a neural network.

## Repository Structure

```text
.
├── README.md
├── METHOD.md
├── DATA.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   └── cross_modal_translation_pipeline.py
└── scripts/
    ├── make_synthetic_h5ad.py
    └── run_demo.sh
```

## Quick Start

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run a synthetic demo:

```bash
bash scripts/run_demo.sh
```

The demo creates small synthetic AnnData files under `data/synthetic/` and writes predictions to `outputs/demo_submission.csv`.

## Running on Real Data

Place local data files under `data/raw/`:

```text
data/raw/adata_rna_train.h5ad
data/raw/adata_met_train.h5ad
data/raw/adata_met_test.h5ad
```

Then run:

```bash
python src/cross_modal_translation_pipeline.py \
  --rna_train data/raw/adata_rna_train.h5ad \
  --met_train data/raw/adata_met_train.h5ad \
  --met_test data/raw/adata_met_test.h5ad \
  --output outputs/submission.csv \
  --model_path models/residual_expression_net.pt \
  --pca_components 512 \
  --hidden_dim 1024 \
  --n_blocks 3 \
  --epochs 120 \
  --batch_size 32 \
  --residual_weight 0.4
```

See [`DATA.md`](DATA.md) for input schema details and [`METHOD.md`](METHOD.md) for the full methodology.

## Notes on Data Availability

This repository does not include the original course or benchmark data, because such datasets may have redistribution restrictions. The repository includes a synthetic-data generator only for testing the code path and demonstrating the expected file format.

## Skills Demonstrated

- Single-cell multi-omics data handling with AnnData.
- Cross-modal prediction from DNA methylation to RNA expression.
- Prototype-based interpretable modeling.
- Residual neural network design with PyTorch.
- Leakage-aware preprocessing and reproducible command-line experiments.

## License

MIT License.
