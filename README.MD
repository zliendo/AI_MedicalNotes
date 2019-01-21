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

### Classification into top level codes in the ICD-9 hierarchy 
| Model | ICD 9 code level| N. Records | Epochs | Notebook |
| --- | --- | --- | --- | --- |
| Baseline | First-Level|5K| -|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br> Section: "Super Basic Baseline with top 4" Always predict top 4 icd-9 codes, F1-score= 52.6|
| CNN Replication| First-Level  | 5K| 20|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br>  Section: "CNN running with 20 epochs". CNN model to replicate results from paper: [Comparing Rule-Based and Deep Learning Models for Patient Phenotyping](https://arxiv.org/abs/1703.08705).In order to compare F1 performance results, I took into consideration the dataset size and number of classes. F1-score= 76.2|  
| CNN| Firs-Level| 5K | 5|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br> Section: "CNN running with 5 epochs" running with the 17 first level ICD-9 codes, using 5 epochs and Embeddings. F1-score= 69.1|
| LSTM | First-Level | 5K| 5|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br>Section "Basic LSTM" running with the 17 first level ICD-9 codes, using 5 epochs and Embeddings. F1-score= 64.6 |
| LSTM with Attention| First-Level | 5K|5| [pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb)|
| CNN-ATT| First-Level | 5K| 5|[pipeline/icd9_cnn_att_workbook.ipynb](pipeline/icd9_cnn_att_workbook.ipynb)|
| Hierarchical LSTM Attention | First-level| 5k|5| [pipeline/icd9_hatt_workbook.ipynb](pipeline/icd9_hatt_workbook.ipynb)
| CNN| First-Level | 5K| 20| [pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb)|

### Classification into most common level 5 ICD-9 Codes
| Model | ICD 9 code level| N. Records | Epochs | Notebook |
| --- | --- | --- | --- | --- |
| Baseline | First-Level | 52.6K| -|[baseline/mimic_icd9_baseline.ipynb](baseline/mimic_icd9_baseline.ipynb) <br/> - Some Initial Exploration with Python and Sql </br> - Basic Baseline Model: For the basic baseline, we make a fixed prediction corresponding to the top 4 ICD-9 codes for all records </br> - NN Baseline Model: A neural network (not Recurrent) with one hidden layer, with relu activation on the hidden layer and sigmoid activation on the output layer. Using cross entropy loss,which is the loss functions for multilabel classification (using Tensorflow)  |
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
