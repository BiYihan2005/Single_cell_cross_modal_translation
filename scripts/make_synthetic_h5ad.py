#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a small synthetic AnnData dataset for pipeline testing.

The synthetic data are designed only to exercise the input/output format and
end-to-end code path. They are not intended to represent real biological data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def make_synthetic_data(
    output_dir: str | Path,
    n_train: int = 300,
    n_test: int = 80,
    n_met_features: int = 600,
    n_genes: int = 120,
    n_types: int = 5,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    type_names = np.array([f"Type_{i}" for i in range(n_types)])
    y_train = rng.integers(0, n_types, size=n_train)
    y_test = rng.integers(0, n_types, size=n_test)

    methylation_centers = rng.normal(0, 1, size=(n_types, n_met_features))
    rna_centers = rng.gamma(shape=2.0, scale=1.0, size=(n_types, n_genes))

    X_met_train = methylation_centers[y_train] + rng.normal(0, 0.8, size=(n_train, n_met_features))
    X_met_test = methylation_centers[y_test] + rng.normal(0, 0.8, size=(n_test, n_met_features))

    cross_modal_weight = rng.normal(0, 0.03, size=(n_met_features, n_genes))
    X_rna_train = (
        rna_centers[y_train]
        + np.maximum(0, X_met_train @ cross_modal_weight)
        + rng.normal(0, 0.15, size=(n_train, n_genes))
    )
    X_rna_train = np.clip(X_rna_train, 0, None)

    train_ids = [f"cell_train_{i:04d}" for i in range(n_train)]
    test_ids = [f"cell_test_{i:04d}" for i in range(n_test)]
    met_var = pd.DataFrame(index=[f"met_feature_{j:04d}" for j in range(n_met_features)])
    rna_var = pd.DataFrame(index=[f"gene_{j:04d}" for j in range(n_genes)])

    met_train = ad.AnnData(
        X=X_met_train.astype(np.float32),
        obs=pd.DataFrame({"MajorType": type_names[y_train]}, index=train_ids),
        var=met_var,
    )
    met_test = ad.AnnData(
        X=X_met_test.astype(np.float32),
        obs=pd.DataFrame(index=test_ids),
        var=met_var.copy(),
    )
    rna_train = ad.AnnData(
        X=X_rna_train.astype(np.float32),
        obs=pd.DataFrame(index=train_ids),
        var=rna_var,
    )

    met_train.write_h5ad(output_dir / "adata_met_train.h5ad")
    met_test.write_h5ad(output_dir / "adata_met_test.h5ad")
    rna_train.write_h5ad(output_dir / "adata_rna_train.h5ad")
    print(f"Synthetic AnnData files written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AnnData files for the demo.")
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_test", type=int, default=80)
    parser.add_argument("--n_met_features", type=int, default=600)
    parser.add_argument("--n_genes", type=int, default=120)
    parser.add_argument("--n_types", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_synthetic_data(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        n_met_features=args.n_met_features,
        n_genes=args.n_genes,
        n_types=args.n_types,
        seed=args.seed,
    )
