#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-cell cross-modal translation from DNA methylation to RNA expression.

The pipeline uses an interpretable prototype-residual strategy:

1. preprocess methylation features and build a PCA representation;
2. infer soft major-cell-type probabilities from methylation features;
3. construct RNA expression prototypes for each major cell type;
4. predict an initial RNA expression profile as a probability-weighted prototype;
5. train a residual neural network to correct cell-level deviations.

Example
-------
python src/cross_modal_translation_pipeline.py \
  --rna_train data/raw/adata_rna_train.h5ad \
  --met_train data/raw/adata_met_train.h5ad \
  --met_test data/raw/adata_met_test.h5ad \
  --output outputs/submission.csv \
  --model_path models/residual_expression_net.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    """Resolve an execution device."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_dense_array(x: Any) -> np.ndarray:
    """Convert an AnnData matrix to a dense NumPy array.

    This implementation favors readability over memory optimization. For very large
    single-cell matrices, consider chunked processing or sparse-aware models.
    """
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory of a path if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_pca_components(requested: int, n_samples: int, n_features: int) -> int:
    """Clip PCA components to a valid range."""
    upper = min(n_samples - 1, n_features)
    if upper < 2:
        raise ValueError(
            f"PCA requires at least two valid components, got n_samples={n_samples}, "
            f"n_features={n_features}."
        )
    return int(max(2, min(requested, upper)))


class ResidualBlock(nn.Module):
    """Feed-forward residual block used in the expression residual network."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResidualExpressionNet(nn.Module):
    """MLP that predicts RNA residuals from methylation PCA features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        n_blocks: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.blocks(h)
        return self.head(h)


def load_and_align_data(
    rna_train_path: str | Path,
    met_train_path: str | Path,
    met_test_path: str | Path,
    major_type_col: str,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Load AnnData files and align cells/features required for supervised training."""
    rna_train = ad.read_h5ad(rna_train_path)
    met_train = ad.read_h5ad(met_train_path)
    met_test = ad.read_h5ad(met_test_path)

    if major_type_col not in met_train.obs.columns:
        raise ValueError(
            f"Column {major_type_col!r} was not found in met_train.obs. "
            f"Available columns: {list(met_train.obs.columns)}"
        )

    common_cells = met_train.obs_names.intersection(rna_train.obs_names)
    if len(common_cells) == 0:
        raise ValueError("RNA training data and methylation training data have no shared cell IDs.")

    common_met_features = met_train.var_names.intersection(met_test.var_names)
    if len(common_met_features) == 0:
        raise ValueError("Methylation train and test matrices have no shared feature IDs.")

    met_train = met_train[common_cells, common_met_features].copy()
    rna_train = rna_train[common_cells, :].copy()
    met_test = met_test[:, common_met_features].copy()

    print(f"Aligned training cells: {met_train.n_obs}")
    print(f"Test cells: {met_test.n_obs}")
    print(f"RNA genes: {rna_train.n_vars}")
    print(f"Shared methylation features: {met_train.n_vars}")

    return rna_train, met_train, met_test


def preprocess_methylation(
    met_train: ad.AnnData,
    met_test: ad.AnnData,
    pca_components: int,
    seed: int,
    preprocess_mode: str = "train_only",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Impute, filter, scale, and PCA-transform methylation features."""
    if preprocess_mode not in {"train_only", "transductive"}:
        raise ValueError("preprocess_mode must be either 'train_only' or 'transductive'.")

    X_train_raw = to_dense_array(met_train.X).astype(np.float32)
    X_test_raw = to_dense_array(met_test.X).astype(np.float32)
    X_fit = np.vstack([X_train_raw, X_test_raw]) if preprocess_mode == "transductive" else X_train_raw

    imputer = SimpleImputer(strategy="mean")
    selector = VarianceThreshold(threshold=0.0)
    scaler = StandardScaler()

    X_fit_imp = imputer.fit_transform(X_fit)
    X_fit_var = selector.fit_transform(X_fit_imp)
    X_fit_scaled = scaler.fit_transform(X_fit_var)

    X_train_scaled = scaler.transform(selector.transform(imputer.transform(X_train_raw)))
    X_test_scaled = scaler.transform(selector.transform(imputer.transform(X_test_raw)))

    n_components = safe_pca_components(pca_components, X_fit_scaled.shape[0], X_fit_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_fit_scaled)

    Z_train = pca.transform(X_train_scaled).astype(np.float32)
    Z_test = pca.transform(X_test_scaled).astype(np.float32)

    summary = {
        "preprocess_mode": preprocess_mode,
        "raw_train_shape": list(X_train_raw.shape),
        "raw_test_shape": list(X_test_raw.shape),
        "n_features_after_variance_filter": int(X_train_scaled.shape[1]),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    print(f"PCA components: {n_components}")
    print(f"Explained variance ratio sum: {summary['pca_explained_variance_ratio_sum']:.4f}")
    return Z_train, Z_test, summary


def build_cell_type_prototypes(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    rna_train: ad.AnnData,
    met_train: ad.AnnData,
    major_type_col: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train a soft cell-type classifier and build RNA prototype predictions."""
    label_encoder = LabelEncoder()
    y_type = label_encoder.fit_transform(met_train.obs[major_type_col].astype(str))

    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    classifier.fit(Z_train, y_type)

    train_prob = classifier.predict_proba(Z_train)
    test_prob = classifier.predict_proba(Z_test)
    Y_train = to_dense_array(rna_train.X).astype(np.float32)

    n_types = len(label_encoder.classes_)
    prototype_expression = np.zeros((n_types, Y_train.shape[1]), dtype=np.float32)
    for type_id in range(n_types):
        mask = y_type == type_id
        if not np.any(mask):
            raise ValueError(f"No training cells found for encoded cell type {type_id}.")
        prototype_expression[type_id] = Y_train[mask].mean(axis=0)

    Y_train_proto = train_prob @ prototype_expression
    Y_test_proto = test_prob @ prototype_expression

    summary = {
        "n_cell_types": int(n_types),
        "cell_types": [str(x) for x in label_encoder.classes_],
        "prototype_shape": list(prototype_expression.shape),
        "mean_max_train_type_probability": float(train_prob.max(axis=1).mean()),
        "mean_max_test_type_probability": float(test_prob.max(axis=1).mean()),
    }
    print(f"Major cell types: {n_types}")
    return Y_train, Y_train_proto.astype(np.float32), Y_test_proto.astype(np.float32), summary


def train_residual_network(
    Z_train: np.ndarray,
    residual_train: np.ndarray,
    hidden_dim: int,
    n_blocks: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    validation_fraction: float,
    patience: int,
    seed: int,
    device_name: str,
    model_path: str | Path,
) -> tuple[ResidualExpressionNet, dict[str, Any]]:
    """Train the residual neural network with early stopping."""
    device = get_device(device_name)
    print(f"Device: {device}")

    Z_train = Z_train.astype(np.float32)
    residual_train = residual_train.astype(np.float32)

    if validation_fraction > 0:
        train_idx, valid_idx = train_test_split(
            np.arange(Z_train.shape[0]),
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_idx = np.arange(Z_train.shape[0])
        valid_idx = np.array([], dtype=int)

    X_train = torch.from_numpy(Z_train[train_idx]).float().to(device)
    Y_train = torch.from_numpy(residual_train[train_idx]).float().to(device)
    if len(valid_idx) > 0:
        X_valid = torch.from_numpy(Z_train[valid_idx]).float().to(device)
        Y_valid = torch.from_numpy(residual_train[valid_idx]).float().to(device)
    else:
        X_valid, Y_valid = X_train, Y_train

    model = ResidualExpressionNet(
        input_dim=Z_train.shape[1],
        output_dim=residual_train.shape[1],
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid_loss = math.inf
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []
    ensure_parent_dir(model_path)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(X_train.size(0), device=device)
        total_loss = 0.0

        for start in range(0, X_train.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = X_train[indices]
            batch_y = Y_train[indices]

            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        train_loss = total_loss / X_train.size(0)
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(X_valid), Y_valid).item()

        history.append({"epoch": epoch, "train_loss": float(train_loss), "valid_loss": float(valid_loss)})
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    summary = {
        "device": str(device),
        "hidden_dim": int(hidden_dim),
        "n_blocks": int(n_blocks),
        "dropout": float(dropout),
        "epochs_requested": int(epochs),
        "epochs_run": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "history": history,
    }
    return model, summary


def predict_residual(
    model: ResidualExpressionNet,
    Z_test: np.ndarray,
    device_name: str,
    batch_size: int,
) -> np.ndarray:
    """Predict residual RNA expression for test cells."""
    device = get_device(device_name)
    model.to(device)
    model.eval()

    X_test = torch.from_numpy(Z_test.astype(np.float32)).float()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_test.size(0), batch_size):
            batch_x = X_test[start : start + batch_size].to(device)
            batch_pred = model(batch_x).cpu().numpy()
            predictions.append(batch_pred)
    return np.vstack(predictions)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the complete cross-modal prediction pipeline."""
    set_seed(args.seed)

    rna_train, met_train, met_test = load_and_align_data(
        rna_train_path=args.rna_train,
        met_train_path=args.met_train,
        met_test_path=args.met_test,
        major_type_col=args.major_type_col,
    )

    Z_train, Z_test, preprocess_summary = preprocess_methylation(
        met_train=met_train,
        met_test=met_test,
        pca_components=args.pca_components,
        seed=args.seed,
        preprocess_mode=args.preprocess_mode,
    )

    Y_train, Y_train_proto, Y_test_proto, prototype_summary = build_cell_type_prototypes(
        Z_train=Z_train,
        Z_test=Z_test,
        rna_train=rna_train,
        met_train=met_train,
        major_type_col=args.major_type_col,
        seed=args.seed,
    )

    residual_train = Y_train - Y_train_proto
    model, training_summary = train_residual_network(
        Z_train=Z_train,
        residual_train=residual_train,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        seed=args.seed,
        device_name=args.device,
        model_path=args.model_path,
    )

    residual_test = predict_residual(
        model=model,
        Z_test=Z_test,
        device_name=args.device,
        batch_size=args.predict_batch_size,
    )
    Y_test_final = Y_test_proto + args.residual_weight * residual_test
    if args.clip_nonnegative:
        Y_test_final = np.clip(Y_test_final, 0, None)

    ensure_parent_dir(args.output)
    submission = pd.DataFrame(Y_test_final, index=met_test.obs_names, columns=rna_train.var_names)
    submission.index.name = "id"
    submission.to_csv(args.output)
    print(f"Prediction saved to: {args.output}")

    summary = {
        "input": {
            "rna_train": str(args.rna_train),
            "met_train": str(args.met_train),
            "met_test": str(args.met_test),
            "major_type_col": args.major_type_col,
        },
        "output": {
            "submission": str(args.output),
            "model_path": str(args.model_path),
        },
        "preprocess": preprocess_summary,
        "prototype": prototype_summary,
        "training": training_summary,
        "fusion": {
            "residual_weight": float(args.residual_weight),
            "clip_nonnegative": bool(args.clip_nonnegative),
        },
    }
    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Run summary saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype-residual cross-modal translation from methylation to RNA expression."
    )
    parser.add_argument("--rna_train", required=True, help="Path to training RNA AnnData file.")
    parser.add_argument("--met_train", required=True, help="Path to training methylation AnnData file.")
    parser.add_argument("--met_test", required=True, help="Path to test methylation AnnData file.")
    parser.add_argument("--output", default="outputs/submission.csv", help="Output CSV path.")
    parser.add_argument("--model_path", default="models/residual_expression_net.pt", help="Model checkpoint path.")
    parser.add_argument("--major_type_col", default="MajorType", help="Cell-type label column in met_train.obs.")
    parser.add_argument(
        "--preprocess_mode",
        default="train_only",
        choices=["train_only", "transductive"],
        help="How to fit unsupervised methylation preprocessing transforms.",
    )
    parser.add_argument("--pca_components", type=int, default=512, help="Number of methylation PCA components.")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension of residual MLP.")
    parser.add_argument("--n_blocks", type=int, default=3, help="Number of residual blocks.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in residual blocks.")
    parser.add_argument("--epochs", type=int, default=120, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--predict_batch_size", type=int, default=128, help="Prediction batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--validation_fraction", type=float, default=0.1, help="Validation fraction for early stopping.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--residual_weight", type=float, default=0.4, help="Weight of neural residual correction.")
    parser.add_argument(
        "--clip_nonnegative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip final expression predictions to non-negative values.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
