# Methodology

This document describes the modeling logic behind the single-cell cross-modal translation pipeline.

## 1. Problem Formulation

Let the methylation matrix be:

$$
X^{\mathrm{met}} \in \mathbb{R}^{n \times d_m},
$$

and the RNA expression matrix be:

$$
Y^{\mathrm{rna}} \in \mathbb{R}^{n \times d_r}.
$$

For each cell $$i$$, the goal is to learn a mapping from methylation features to RNA expression:

$$
f: X^{\mathrm{met}}_i \mapsto Y^{\mathrm{rna}}_i.
$$

The challenge is that $$d_r$$ can be large, the two modalities are not directly aligned feature by feature, and single-cell measurements often contain strong technical noise.

## 2. Key Modeling Assumption

The pipeline is based on the assumption that RNA expression variation can be decomposed into two components:

$$
Y_i
\approx
\text{cell-type-level expression}
+
\text{cell-level residual variation}.
$$

The first component is captured by RNA prototypes for major cell types. The second component is modeled by a residual neural network using methylation-derived representations.

## 3. Methylation Representation

The methylation feature matrix is first preprocessed by:

1. mean imputation for missing values;
2. zero-variance feature filtering;
3. standard scaling;
4. PCA dimensionality reduction.

This produces a low-dimensional methylation representation:

$$
Z = \mathrm{PCA}(\mathrm{scale}(\mathrm{filter}(\mathrm{impute}(X^{\mathrm{met}})))).
$$

The implementation supports two preprocessing modes:

- `train_only`: fit preprocessing transforms on training methylation data only, then apply them to test data. This is the strict no-leakage setting.
- `transductive`: fit unsupervised preprocessing transforms on the combined train and test methylation matrix. This can be useful in competition-style transductive settings, but it should be reported explicitly.

## 4. Soft Cell-Type Inference

A classifier is trained on methylation representations to estimate major cell-type probabilities:

$$
P_i = \Pr(c_i = k \mid Z_i), \quad k = 1, \dots, K.
$$

For all test cells, the classifier outputs:

$$
P_{\mathrm{test}} \in \mathbb{R}^{n_{\mathrm{test}} \times K}.
$$

Soft probabilities are preferred over hard labels because single-cell states can be continuous, transitional, or uncertain. A probability vector preserves this uncertainty and allows multiple cell-type prototypes to contribute to the predicted expression profile.

## 5. RNA Prototype Construction

For each major cell type $k$, define the set of training cells belonging to that type as $S_k$. The RNA prototype for type $k$ is:

$$
\begin{aligned}
C_k
&=
\frac{1}{|S_k|}
\sum_{i \in S_k} Y_i .
\end{aligned}
$$

Stacking all prototypes gives:

$$
C \in \mathbb{R}^{K \times d_r}.
$$

For a cell with soft type probability vector $P_i$, the prototype-based RNA prediction is:

$$
\begin{aligned}
\widehat{Y}^{\mathrm{proto}}_i
&=
P_i C .
\end{aligned}
$$

This term captures the dominant transcriptional pattern explained by cell identity.

## 6. Residual Neural Correction

The prototype term is robust but cannot fully explain intra-type heterogeneity. Therefore, the training residual is defined as:

$$
\begin{aligned}
R_i
&=
Y_i
-
\widehat{Y}^{\mathrm{proto}}_i .
\end{aligned}
$$

A neural network $g_{\theta}$ is trained to predict this residual from methylation PCA features:

$$
\begin{aligned}
g_{\theta}(Z_i)
&\approx
R_i .
\end{aligned}
$$

The residual network is a multilayer perceptron with residual blocks. A residual block has the general form:

$$
\begin{aligned}
h_{\ell+1}
&=
\sigma\left(
h_{\ell}
+
F_{\ell}(h_{\ell})
\right) .
\end{aligned}
$$

where $F_{\ell}$ is a small feed-forward transformation and $\sigma$ is a nonlinear activation function. Residual connections make optimization more stable and match the modeling goal of learning corrections rather than reconstructing expression from scratch.

## 7. Training Objective

The residual network is trained with mean squared error:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{res}}
&=
\frac{1}{n}
\sum_{i=1}^{n}
\left\|
g_{\theta}(Z_i) - R_i
\right\|_2^2 .
\end{aligned}
$$

Early stopping is applied using a validation split to reduce overfitting. The model checkpoint with the best validation loss is used for test prediction.

## 8. Final Prediction

For test cells, the final prediction is:

$$
\begin{aligned}
\widehat{Y}_{\mathrm{test}}
&=
P_{\mathrm{test}}C
+
\alpha g_{\theta}(Z_{\mathrm{test}}) .
\end{aligned}
$$

where $\alpha$ controls the strength of residual correction.

Because RNA expression is non-negative, the final prediction can be clipped by:

$$
\begin{aligned}
\widehat{Y}_{\mathrm{test}}
&\leftarrow
\max\left(
\widehat{Y}_{\mathrm{test}}, 0
\right) .
\end{aligned}
$$

## 9. Interpretation

The prediction can be interpreted as follows:

- $$P_{\mathrm{test}}C$$ provides a stable cell-type-level expression baseline.
- $$g_{\theta}(Z_{\mathrm{test}})$$ adds methylation-informed cell-level deviations.
- $$\alpha$$ controls the bias-variance trade-off between conservative prototype prediction and flexible neural correction.

When $$\alpha = 0$$, the model reduces to a purely prototype-based predictor. As $$\alpha$$ increases, the residual network contributes more strongly.

## 10. Limitations and Future Directions

This repository is intentionally compact and focuses on a clean, interpretable baseline. Potential extensions include:

- out-of-fold prototype probabilities for stricter residual training;
- sparse or chunked processing for large-scale single-cell matrices;
- gene-level biological priors linking methylation regions to target genes;
- variational or contrastive latent-space alignment;
- graph-based modeling of cell neighborhoods;
- validation-based tuning of the residual weight $$\alpha$$.
