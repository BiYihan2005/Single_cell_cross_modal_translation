# 数据格式说明

本项目默认输入文件为 AnnData `.h5ad` 格式。

## 1. `adata_rna_train.h5ad`

训练集 RNA 表达数据。

- `adata.X`：RNA 表达矩阵；
- `adata.obs_names`：训练细胞 id；
- `adata.var_names`：基因名称。

要求：`obs_names` 能够与 `adata_met_train.h5ad` 中的训练细胞 id 对齐。

## 2. `adata_met_train.h5ad`

训练集 DNA 甲基化数据。

- `adata.X`：甲基化特征矩阵；
- `adata.obs_names`：训练细胞 id；
- `adata.var_names`：甲基化特征名称；
- `adata.obs["MajorType"]`：细胞大类标签。

如果你的标签列不叫 `MajorType`，可以通过命令行参数指定：

```bash
--major_type_col cell_type
```

## 3. `adata_met_test.h5ad`

测试集 DNA 甲基化数据。

- `adata.X`：测试细胞甲基化特征；
- `adata.obs_names`：测试细胞 id；
- `adata.var_names`：应与训练甲基化数据的特征空间一致。

## 4. 输出文件格式

输出 `submission.csv`：

```text
id,gene_1,gene_2,gene_3,...
cell_test_0001,0.13,0.00,1.52,...
cell_test_0002,0.05,0.71,0.00,...
```

行是测试细胞，列是 RNA 基因。
