# Classifying medical notes into standard disease codes

**August 2017**

This repository contains the code I implemented to classify automatically EHR patient discharge notes into standard
disease labels (ICD9 codes). I implemented deep learning models ( CNN, LSTM and Hierarchical models) using embeddings and 
attention layers. The CNN model with attention outperformed previous algorithms used in this task.   
The dataset used for modeling was: [MIMIC III dataset](https://mimic.physionet.org) .

The code was implemented on August 2017, during my graduate studies at the Master of Information and Data Science (MIDS) program at UC Berkeley. The class was: W266 Natural Language Processing with Deep Learning   

This is the final project report: [w266FinalReport_ICD_9_Classification.pdf](w266FinalReport_ICD_9_Classification.pdf)

(note: code refactoring pending)

## Main Notebooks
| Model | Level| N. Records | Epochs | Notebook |
| --- | --- | --- | --- | --- |
| Baseline | First-Level | 52.6K| -|[baseline/mimic_icd9_baseline.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) <br/> - Some Initial Exploration with Python and Sql </br> - Basic Baseline Model: For the basic baseline, we make a fixed prediction corresponding to the top 4 ICD-9 codes for all records </br> - NN Baseline Model: A neural network (not Recurrent) with one hidden layer, with relu activation on the hidden layer and sigmoid activation on the output layer. Using cross entropy loss,which is the loss functions for multilabel classification (using Tensorflow)  |
| LSTM | First-Level | 5K| 5|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb)|
| LSTM with Attention| First-Level | 5K|5| [pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb)|
| CNN| First-Level | 5K| 5|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb)|
| CNN-ATT| First-Level | 5K| 5|[pipeline/icd9_cnn_att_workbook.ipynb](pipeline/icd9_cnn_att_workbook.ipynb)|
| Hierarchical LSTM Attention | First-level| 5k|5| [pipeline/icd9_hatt_workbook.ipynb](pipeline/icd9_hatt_workbook.ipynb)
| CNN| First-Level | 5K| 20| [pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb)|
| Basic Baseline | Leaf | 46K | - | [baseline/mimic_icd9_baseline.ipynb](baseline/mimic_icd9_baseline.ipynb) |
| CNN for top 20 leaf icd-9 codes | Leaf | 46K | 7 | [icd9_cnn/cnn_top20_leave.ipynb](icd9_cnn/cnn_top20_leave.ipynb) |
| CNN for first-level icd-9 codes in hierarchy | First-Level | 52.6K | - | [pipeline/icd9_cnn_50K_run.ipynb](/pipeline/icd9_cnn_50K_run.ipynb) |
| CNN-ATT for first-level icd-9 codes in hierarchy | First-Level | 52.6K | - | [pipeline/icd9_cnn_att_50K_records.ipynb](pipeline/icd9_cnn_att_50K_records.ipynb) |



## Model Python modules

| Model | Python module |
| --- | --- |
| LSTM | [pipeline/lstm_model.py]pipeline/lstm_model.py) |
| CNN | [pipeline/icd9_cnn_model.py](pipeline/icd9_cnn_model.py)  |
| Attention Layer |[pipeline/attention_util.py](pipeline/attention_util.py)  |
| LSTM_ATT | [pipeline/icd9_lstm_att_model.py](pipeline/icd9_lstm_att_model.py)   |
| CNN_ATT | [pipeline/icd9_cnn_att.py](pipeline/icd9_cnn_att.py)   |
| Hierarchical LSTM Attention | [pipeline/hatt_model.py](pipeline/hatt_model.py)  |
