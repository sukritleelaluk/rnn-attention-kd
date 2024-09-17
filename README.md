# Overview

This repository contains supplementary materials for the following conference paper:

Anonymous author(s).
Early Prediction of Student Performance Using Knowledge Distillation RNN-Attention Models
Submitted to the 40th ACM/SIGAPP Symposium On Applied Computing (SAC 2025).

# File description

### `1_Preprocessing.ipynb`
Convert EventStream log into Text file.\
See Section 3.2 in the paper.

### `2_train_fastText.ipynb`
Train fastText with preprocessed text.\
See Section 3.3 in the paper.

### `3_Making_CodeBook.ipynb`
Make CodeBook for Aggregation.
Perform k-means++ clustering for action vectors.\
See Section 3.4.1 in the paper.

### `4_Embedding_and_Aggregation.ipynb`
Generate students vector in one lecture course.

### `5_At-risk_prediction.ipynb`
Predict students grade with student vectors.\
See Section 7 in the paper.
