# 单细胞跨模态翻译：基于细胞类型原型与残差神经网络的 RNA 表达预测

本项目整理自一次单细胞多组学预测实验，目标是根据单细胞 DNA 甲基化模态预测 RNA 表达模态。项目将原本的一次性实验脚本重构为可复用的研究型 pipeline，并补充中文文档、方法说明和合成数据 demo，便于理解、复现和二次开发。

## 项目亮点

- **跨模态预测任务**：输入 scDNA methylation 特征，输出 scRNA expression 矩阵。
- **两阶段可解释建模**：先用甲基化数据预测细胞类型概率，再基于 RNA 原型表达进行初步预测。
- **残差神经网络修正**：用深度残差网络学习原型表达无法解释的细胞个体差异。
- **可复用命令行 pipeline**：支持自定义 PCA 维度、残差权重、网络深度、训练轮数等参数。
- **不公开真实课程数据**：仓库只提供合成 demo 数据生成脚本，真实数据请自行放入 `data/raw/`。

## 方法概览

## 方法概览

本项目的核心预测公式为：

P_{\mathrm{test}} C
+
\alpha \cdot g_{\theta}(Z_{\mathrm{test}})
$$

其中：

* $P_{\mathrm{test}}$：测试细胞属于各个 MajorType 的概率矩阵；
* $C$：训练集中各 MajorType 的平均 RNA 表达原型；
* $Z_{\mathrm{test}}$：测试甲基化数据经过预处理和 PCA 后得到的低维表示；
* $g_{\theta}$：残差神经网络，用于学习原型表达无法解释的个体差异；
* $\alpha$：残差融合权重，默认取 0.4。

直观理解是：模型先根据 DNA 甲基化特征判断测试细胞更像哪些细胞类型，再用这些细胞类型的 RNA 表达原型生成基础预测，最后用残差神经网络对基础预测进行修正。

## 仓库结构

```text
single-cell-cross-modal-translation/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── .gitattributes
├── src/
│   └── cross_modal_translation_pipeline.py
├── scripts/
│   ├── make_synthetic_h5ad.py
│   └── run_demo.sh
├── data/
│   ├── README.md
│   ├── raw/
│   └── example/
├── docs/
│   ├── data_schema.md
│   ├── methodology.md
│   └── github_upload_guide.md
├── reports/
│   ├── project_summary.md
│   └── algorithm_design_notes_zh.md
├── legacy/
│   └── run_task_nn_submission_original.py
├── outputs/
├── models/
└── notebooks/
```

## 快速开始

### 1. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 运行合成数据 demo

```bash
bash scripts/run_demo.sh
```

运行结束后会生成：

```text
outputs/demo_submission.csv
models/demo_residual_net.pt
outputs/demo_submission.summary.json
```

### 3. 使用真实数据运行

将真实数据放到 `data/raw/`：

```text
data/raw/adata_rna_train.h5ad
data/raw/adata_met_train.h5ad
data/raw/adata_met_test.h5ad
```

然后运行：

```bash
python3 src/cross_modal_translation_pipeline.py \
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

## 输入数据要求

训练集需要三个 AnnData 文件：

1. `adata_rna_train.h5ad`
   - 行：训练细胞；
   - 列：RNA 基因；
   - `obs_names` 需要与训练甲基化数据中的细胞 id 对齐。

2. `adata_met_train.h5ad`
   - 行：训练细胞；
   - 列：甲基化特征；
   - `obs` 中需要包含 `MajorType` 列。

3. `adata_met_test.h5ad`
   - 行：测试细胞；
   - 列：甲基化特征；
   - 不需要 RNA 标签。

更多细节见 [`docs/data_schema.md`](docs/data_schema.md)。

## 为什么不上传真实数据

真实课程数据或竞赛数据通常存在再分发限制。为了避免数据版权和隐私问题，本仓库不包含真实原始数据，只提供合成数据脚本用于验证代码流程。真实数据请自行下载或按课程/竞赛要求放入本地 `data/raw/`。该目录已被 `.gitignore` 忽略。

## 依赖

主要依赖包括：

- numpy / pandas
- scanpy / anndata
- scikit-learn
- PyTorch

详见 `requirements.txt`。

## 项目定位

本项目适合作为单细胞多组学跨模态预测的入门级研究 pipeline。它不追求覆盖所有最先进方法，而是强调：

- 任务拆解；
- 可解释建模；
- 代码可读性；
- 可复现运行；
- 从一次性脚本到研究项目的整理过程。

## License

MIT License.
