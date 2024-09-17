# Overview

This repository contains supplementary materials for the following conference paper:

Anonymous author(s).
Early Prediction of Student Performance Using Knowledge Distillation RNN-Attention Models
Submitted to the 40th ACM/SIGAPP Symposium On Applied Computing (SAC 2025).

# File description

### `utils/rnn_attention_kd.py`
Show the structure of RNN-Attention-KD\
See Section 3.2 in the paper.

### `01_teacher_model_training.ipynb`
Training the teacher model using the entire course data to prepare for distilling knowledge to the student model.\
See Section 3.2 and 4.2.2 in the paper.

### `02_student_model_training.ipynb`
Training the student model using the knowledge from the teacher model.\
See Section 3.2 and 4.2.2 in the paper.
