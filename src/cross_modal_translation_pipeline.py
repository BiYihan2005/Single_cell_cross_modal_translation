#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单细胞跨模态翻译 pipeline：由 DNA 甲基化模态预测 RNA 表达模态。

项目背景
--------
本项目面向单细胞多组学跨模态预测任务。训练集中同时给出：
1. scDNA methylation（甲基化）数据；
2. scRNA expression（RNA 表达）数据；
3. 训练细胞的 MajorType 标签。

测试集中只给出甲基化数据。模型需要根据测试细胞的甲基化特征预测其 RNA 表达矩阵。

核心思想
--------
直接用一个神经网络从甲基化矩阵端到端预测 RNA 表达非常困难，因为输出维度高、噪声大、
细胞类型差异强。为降低学习难度，本脚本采用“两阶段”思路：

第一阶段：细胞类型软分类 + RNA 原型表达
    使用甲基化特征预测细胞属于各 MajorType 的概率；
    再用这些概率对各细胞类型的平均 RNA 表达进行加权平均，得到一个稳定的初步预测。

第二阶段：残差神经网络修正
    计算训练集真实 RNA 与“原型预测”的差值，即残差；
    用残差神经网络学习甲基化特征中尚未被细胞类型原型解释的细节信息；
    最终预测 = 原型预测 + residual_weight × 残差预测。

相比直接端到端回归，该方法更可解释：
- 原型表达捕捉细胞类型主导的共性表达模式；
- 残差网络捕捉细胞内部状态、连续过渡和个体差异。

输入文件
--------
--rna_train: 训练集 RNA AnnData 文件，要求 obs_names 与训练甲基化细胞可对齐。
--met_train: 训练集甲基化 AnnData 文件，obs 中需包含 MajorType 标签。
--met_test:  测试集甲基化 AnnData 文件。

输出文件
--------
submission.csv:
    行为测试细胞 id，列为 RNA 基因，值为预测表达量。

training_summary.json:
    保存主要参数、类别数量、PCA 解释方差、训练损失等摘要信息。

运行示例
--------
python src/cross_modal_translation_pipeline.py \
  --rna_train data/raw/adata_rna_train.h5ad \
  --met_train data/raw/adata_met_train.h5ad \
  --met_test data/raw/adata_met_test.h5ad \
  --output outputs/submission.csv \
  --epochs 80 \
  --pca_components 256 \
  --residual_weight 0.4
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

import scanpy as sc
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# 1. 通用工具函数
# =============================================================================

def set_seed(seed: int) -> None:
    """固定随机种子，提高结果可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_dense_array(x):
    """
    将 AnnData 中的稀疏矩阵转换为 dense numpy array。

    注意：
    单细胞数据通常非常稀疏，直接转 dense 可能占用大量内存。
    本项目为了保持代码清晰和教学可读性，采用 dense 形式处理。
    如果数据规模更大，建议进一步改成分块处理或使用稀疏矩阵友好的模型。
    """
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def ensure_parent_dir(path: str | Path) -> None:
    """确保输出文件的父目录存在。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_pca_components(requested: int, n_samples: int, n_features: int) -> int:
    """
    自动修正 PCA 维度，避免 n_components 超过样本数或特征数。
    PCA 的最大可用维度通常不能超过 min(n_samples, n_features)。
    """
    return int(max(2, min(requested, n_samples - 1, n_features)))


# =============================================================================
# 2. 模型定义：残差块与深度残差网络
# =============================================================================

class ResidualBlock(nn.Module):
    """
    简单残差块。

    残差连接的形式是：
        output = F(x) + x

    这样做的好处是：
    - 缓解深层网络训练中的梯度消失；
    - 让网络更容易学习“相对于输入的修正项”；
    - 在残差预测任务中尤其自然。
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class ResidualExpressionNet(nn.Module):
    """
    用于预测 RNA 残差的神经网络。

    输入：
        PCA 后的甲基化低维表征 Z。

    输出：
        RNA 表达残差矩阵的一行，即某个细胞在所有基因上的残差预测。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        n_blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.fc_out(x)
        return x


# =============================================================================
# 3. 预处理与原型表达构建
# =============================================================================

def load_and_align_data(
    rna_train_path: str | Path,
    met_train_path: str | Path,
    met_test_path: str | Path,
    major_type_col: str,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """
    读取 AnnData 数据，并将训练集 RNA 与训练集甲基化按共同细胞对齐。

    为什么要对齐？
    RNA 与甲基化是同一批训练细胞的两个模态，只有 cell id 对齐后，
    模型才能学习“这个细胞的甲基化特征 -> 这个细胞的 RNA 表达”。
    """

    print("读取 AnnData 文件...")
    rna_train = sc.read_h5ad(rna_train_path)
    met_train = sc.read_h5ad(met_train_path)
    met_test = sc.read_h5ad(met_test_path)

    if major_type_col not in met_train.obs.columns:
        raise ValueError(
            f"met_train.obs 中找不到标签列 {major_type_col!r}。"
            f"当前可用列为：{list(met_train.obs.columns)}"
        )

    common_cells = met_train.obs_names.intersection(rna_train.obs_names)
    if len(common_cells) == 0:
        raise ValueError("训练集 RNA 与甲基化数据没有共同细胞 id，无法对齐。")

    met_train = met_train[common_cells].copy()
    rna_train = rna_train[common_cells].copy()

    print(f"训练细胞数：{met_train.n_obs}")
    print(f"测试细胞数：{met_test.n_obs}")
    print(f"RNA 基因数：{rna_train.n_vars}")
    print(f"甲基化特征数：{met_train.n_vars}")

    return rna_train, met_train, met_test


def preprocess_methylation(
    met_train: ad.AnnData,
    met_test: ad.AnnData,
    pca_components: int,
    seed: int,
    preprocess_mode: str = "train_only",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    预处理甲基化数据。

    步骤：
    1. 缺失值均值填补；
    2. 删除零方差特征；
    3. 标准化；
    4. PCA 降维。

    preprocess_mode 说明：
    - train_only：只在训练集上拟合 imputer / selector / scaler / PCA，再应用到测试集。
      这是最标准的无泄漏做法。
    - transductive：训练集和测试集的甲基化 X 合并后一起拟合无监督预处理参数。
      这种做法不使用测试集 RNA 标签，在部分竞赛场景中可接受，但严格泛化研究中建议用 train_only。
    """

    print("预处理甲基化数据...")
    X_train_raw = to_dense_array(met_train.X)
    X_test_raw = to_dense_array(met_test.X)

    if preprocess_mode not in {"train_only", "transductive"}:
        raise ValueError("preprocess_mode 只能是 'train_only' 或 'transductive'。")

    if preprocess_mode == "transductive":
        X_fit = np.vstack([X_train_raw, X_test_raw])
    else:
        X_fit = X_train_raw

    imputer = SimpleImputer(strategy="mean")
    selector = VarianceThreshold(threshold=0.0)
    scaler = StandardScaler()

    X_fit_imp = imputer.fit_transform(X_fit)
    X_fit_var = selector.fit_transform(X_fit_imp)
    X_fit_scaled = scaler.fit_transform(X_fit_var)

    X_train_scaled = scaler.transform(selector.transform(imputer.transform(X_train_raw)))
    X_test_scaled = scaler.transform(selector.transform(imputer.transform(X_test_raw)))

    n_components = safe_pca_components(
        pca_components,
        n_samples=X_fit_scaled.shape[0],
        n_features=X_fit_scaled.shape[1],
    )

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_fit_scaled)

    Z_train = pca.transform(X_train_scaled)
    Z_test = pca.transform(X_test_scaled)

    summary = {
        "preprocess_mode": preprocess_mode,
        "raw_train_shape": list(X_train_raw.shape),
        "raw_test_shape": list(X_test_raw.shape),
        "n_features_after_variance_filter": int(X_train_scaled.shape[1]),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }

    print(f"PCA 维度：{n_components}")
    print(f"PCA 累计解释方差比例：{summary['pca_explained_variance_ratio_sum']:.4f}")

    return Z_train, Z_test, summary


def build_cell_type_prototypes(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    rna_train: ad.AnnData,
    met_train: ad.AnnData,
    major_type_col: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, Dict]:
    """
    构建“细胞类型软分类 + 原型表达”预测。

    具体做法：
    1. 用甲基化 PCA 特征训练逻辑回归，预测 MajorType；
    2. 用 predict_proba 得到每个细胞属于各类型的概率；
    3. 对训练集中每个 MajorType 计算平均 RNA 表达，形成类型原型；
    4. 用概率矩阵 × 原型矩阵得到 RNA 初步预测。

    为什么使用 predict_proba 而不是 predict？
    因为单细胞状态可能存在连续过渡。概率向量能保留“一个细胞像多个类型”的信息，
    比硬分类更柔和、更稳健。
    """

    print("训练 MajorType 软分类器，并构建 RNA 原型表达...")

    label_encoder = LabelEncoder()
    y_type = label_encoder.fit_transform(met_train.obs[major_type_col].astype(str))

    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
        n_jobs=None,
    )
    classifier.fit(Z_train, y_type)

    train_prob = classifier.predict_proba(Z_train)
    test_prob = classifier.predict_proba(Z_test)

    Y_train = to_dense_array(rna_train.X).astype(np.float32)

    n_types = len(label_encoder.classes_)
    proto_expr = np.zeros((n_types, Y_train.shape[1]), dtype=np.float32)

    for c in range(n_types):
        mask = (y_type == c)
        proto_expr[c] = Y_train[mask].mean(axis=0)

    Y_train_proto = train_prob @ proto_expr
    Y_test_proto = test_prob @ proto_expr

    summary = {
        "n_cell_types": int(n_types),
        "cell_types": [str(x) for x in label_encoder.classes_],
        "prototype_shape": list(proto_expr.shape),
    }

    print(f"MajorType 类别数：{n_types}")

    return Y_train, Y_train_proto, Y_test_proto, label_encoder, summary


# =============================================================================
# 4. 残差网络训练与预测
# =============================================================================

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
) -> Tuple[ResidualExpressionNet, Dict]:
    """
    训练残差网络。

    训练目标：
        输入甲基化 PCA 特征 Z；
        输出 RNA 残差 residual = 真实 RNA - 原型预测 RNA。

    使用早停策略：
        如果验证集 MSE 连续 patience 轮没有改善，则提前停止训练。
    """

    print("训练残差神经网络...")

    device = torch.device(
        device_name if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"使用设备：{device}")

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

    X_tr = torch.from_numpy(Z_train[train_idx]).float().to(device)
    Y_tr = torch.from_numpy(residual_train[train_idx]).float().to(device)

    if len(valid_idx) > 0:
        X_va = torch.from_numpy(Z_train[valid_idx]).float().to(device)
        Y_va = torch.from_numpy(residual_train[valid_idx]).float().to(device)
    else:
        X_va = X_tr
        Y_va = Y_tr

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
    history = []

    ensure_parent_dir(model_path)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(X_tr.size(0), device=device)
        total_loss = 0.0

        for start in range(0, X_tr.size(0), batch_size):
            idx = permutation[start:start + batch_size]
            batch_x = X_tr[idx]
            batch_y = Y_tr[idx]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        train_loss = total_loss / X_tr.size(0)

        model.eval()
        with torch.no_grad():
            valid_pred = model(X_va)
            valid_loss = criterion(valid_pred, Y_va).item()

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
        })

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs}, train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"验证集损失连续 {patience} 轮未改善，提前停止。")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
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
    """用训练好的残差网络预测测试集残差。"""

    device = torch.device(
        device_name if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model.to(device)
    model.eval()

    X = torch.from_numpy(Z_test.astype(np.float32)).float()
    preds = []

    with torch.no_grad():
        for start in range(0, X.size(0), batch_size):
            batch_x = X[start:start + batch_size].to(device)
            batch_pred = model(batch_x).cpu().numpy()
            preds.append(batch_pred)

    return np.vstack(preds)


# =============================================================================
# 5. 主流程
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """按顺序执行完整跨模态预测流程。"""

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

    Y_train, Y_train_proto, Y_test_proto, label_encoder, proto_summary = build_cell_type_prototypes(
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

    # 融合预测：原型表达负责稳定主体结构，残差网络负责补充细节。
    Y_test_final = Y_test_proto + args.residual_weight * residual_test

    # RNA 表达量通常非负，因此最后将负值截断为 0。
    if args.clip_nonnegative:
        Y_test_final = np.clip(Y_test_final, 0, None)

    ensure_parent_dir(args.output)
    submission = pd.DataFrame(
        Y_test_final,
        index=met_test.obs_names,
        columns=rna_train.var_names,
    )
    submission.index.name = "id"
    submission.to_csv(args.output)
    print(f"预测结果已保存：{args.output}")

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
        "prototype": proto_summary,
        "training": training_summary,
        "fusion": {
            "residual_weight": float(args.residual_weight),
            "clip_nonnegative": bool(args.clip_nonnegative),
        },
    }

    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"训练摘要已保存：{summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单细胞甲基化到 RNA 表达的跨模态预测 pipeline")

    # 输入输出路径
    parser.add_argument("--rna_train", required=True, help="训练集 RNA h5ad 文件路径")
    parser.add_argument("--met_train", required=True, help="训练集甲基化 h5ad 文件路径")
    parser.add_argument("--met_test", required=True, help="测试集甲基化 h5ad 文件路径")
    parser.add_argument("--output", default="outputs/submission.csv", help="预测结果 CSV 输出路径")
    parser.add_argument("--model_path", default="models/residual_expression_net.pt", help="残差网络模型保存路径")

    # 数据字段
    parser.add_argument("--major_type_col", default="MajorType", help="met_train.obs 中表示细胞大类的列名")

    # 预处理参数
    parser.add_argument("--preprocess_mode", default="train_only", choices=["train_only", "transductive"],
                        help="预处理参数拟合方式：train_only 更严格；transductive 可复现竞赛式做法")
    parser.add_argument("--pca_components", type=int, default=512, help="PCA 降维后的维度")

    # 神经网络参数
    parser.add_argument("--hidden_dim", type=int, default=1024, help="残差网络隐藏层维度")
    parser.add_argument("--n_blocks", type=int, default=3, help="残差块数量")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout 比例")
    parser.add_argument("--epochs", type=int, default=120, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="训练 batch size")
    parser.add_argument("--predict_batch_size", type=int, default=128, help="预测 batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW 权重衰减")
    parser.add_argument("--validation_fraction", type=float, default=0.1, help="残差网络验证集比例")
    parser.add_argument("--patience", type=int, default=20, help="早停 patience")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")

    # 融合与后处理参数
    parser.add_argument("--residual_weight", type=float, default=0.4, help="残差预测融合权重")
    parser.add_argument("--clip_nonnegative", action="store_true", default=True, help="是否将预测表达量截断为非负")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
