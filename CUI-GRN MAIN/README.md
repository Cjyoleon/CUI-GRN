# CUI-GRN: a framework for single-cell gene regulatory network inference integrating complementary candidate support, pseudo-negative ensembling, and intermediate-layer representation reuse

## Overview

CUI-GRN is a framework for inferring gene regulatory networks from single-cell RNA-seq data. It integrates:

- Complementary Candidate Support Residual Adapter (CCSRA)
- Uniform Pseudo-Negative Ensembling (UPNE) strategy
- Intermediate-Layer Representation Reuse (ILRR)

## Abstract
Single-cell RNA sequencing enables inference of gene regulatory networks (GRNs), yet incomplete regulatory annotations restrict the structural information available to supervised models. Moreover, using a single negative set sampled from unlabeled transcription factor (TF)–target pairs may make predictions sensitive to its composition. To address these challenges, we propose CUI-GRN, a framework integrating complementary candidate support, pseudo-negative ensembling, and intermediate-layer representation reuse. The complementary candidate support residual adapter (CCSRA) uses complementary motif-derived candidate interactions and their expression-supported subset to refine gene representations through auxiliary neighborhood residuals without changing the main propagation graph constructed from positive training edges. In configurations where different pseudo-negative sets can be sampled under fixed per-TF negative-sample budgets, uniform pseudo-negative ensembling (UPNE) averages predictions from models trained on independently resampled sets to reduce dependence on a single sampled set. Intermediate-layer representation reuse (ILRR) supplements the final encoding with intermediate-layer representations through learnable cross-layer residual fusion. Evaluations across 44 benchmark configurations spanning four network types yielded average AUPRC and AUROC values of 0.43019 and 0.83354, respectively. CUI-GRN achieved performance comparable to or better than existing methods, with more pronounced improvements in sparse-network configurations. Ablation experiments further supported the roles of the three modules within the complete framework. Case studies provided additional evidence for the biological relevance of CUI-GRN predictions. These results suggest that CUI-GRN can improve single-cell GRN inference under incomplete regulatory annotations.

## Usage

CUI-GRN requires single-cell RNA-seq expression data in the following format:

Input: `G × N` expression matrix (G genes × N cells)

**Command to run model**

```bash
python main.py --epochs 30 --batch_size 256 --net Non-Specific --num 1000 --data mHSC-GM
```

## Contact

For questions, contact: yuliyang_0@126.com