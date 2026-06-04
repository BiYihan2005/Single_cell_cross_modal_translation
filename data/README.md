# 数据目录说明

本仓库不包含真实课程数据或竞赛数据。

## 推荐目录结构

```text
data/
├── raw/
│   ├── adata_rna_train.h5ad
│   ├── adata_met_train.h5ad
│   └── adata_met_test.h5ad
└── example/
    └── 由 scripts/make_synthetic_h5ad.py 生成的合成数据
```

## 为什么不上传真实数据

真实单细胞多组学数据通常存在课程、竞赛或数据平台的使用限制。为了避免数据再分发问题，本仓库只公开代码和文档。

如果你有合法获得的数据，请将其放入 `data/raw/`。该目录已被 `.gitignore` 忽略，不会被提交到 GitHub。
