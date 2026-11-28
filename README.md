# CUI-GRN: a framework for single-cell gene regulatory network inference integrating complementary candidate support, pseudo-negative ensembling, and intermediate-layer representation reuse

## Overview

CUI-GRN is a framework for inferring gene regulatory networks from single-cell RNA-seq data. 

## Usage

CUI-GRN requires single-cell RNA-seq expression data in the following format:

Input: `G × N` expression matrix (G genes × N cells)

**Command to run model**

```bash
python main.py --epochs 30 --batch_size 256 --net Non-Specific --num 1000 --data mHSC-GM
```

## Contact

For questions, contact: yuliyang_0@126.com
