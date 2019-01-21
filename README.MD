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

### Classification into top level codes in the ICD-9 hierarchy with 5K records
| Model | ICD 9 code level| N. Records | Epochs | Notebook |
| --- | --- | --- | --- | --- |
| Baseline | First-Level|5K| -|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br> Section: "Super Basic Baseline with top 4" Always predict top 4 icd-9 codes, F1-score= 52.6|
| CNN Replication| First-Level  | 5K| 20|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br>  Section: "CNN running with 20 epochs". CNN model to replicate results from paper: [Comparing Rule-Based and Deep Learning Models for Patient Phenotyping](https://arxiv.org/abs/1703.08705).In order to compare F1 performance results, I took into consideration the dataset size and number of classes. F1-score= 76.2|  
| CNN| Firs-Level| 5K | 5|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br> Section: "CNN running with 5 epochs" running with the 17 first level ICD-9 codes, using 5 epochs and Embeddings. F1-score= 69.1|
| LSTM | First-Level | 5K| 5|[pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br>Section "Basic LSTM" running with the 17 first level ICD-9 codes, using 5 epochs and Embeddings. F1-score= 64.6 |

**Attention**   
The average length of discharge clinical notes is 1639 words. The text to classify may be too long for a LSTM or CNN to
remember all relevant information. [Raffel et al. (2016)](https://arxiv.org/abs/1512.08756) displayed better performance in many NLP tasks on long text using Attention. Here, we seek to emulate his results by implementing algorithms based on the formulas presented in [Raffel et al. (2016)](https://arxiv.org/abs/1512.08756) and [Yang et al. (2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).   
   
| Model | ICD 9 code level| N. Records | Epochs | Notebook |
| --- | --- | --- | --- | --- |
| LSTM with Attention| First-Level | 5K|5| [pipeline/icd9_lstm_cnn_workbook.ipynb](pipeline/icd9_lstm_cnn_workbook.ipynb) </br> Section: "LSTM with Attention"</br> F1-score:67.0|
| CNN with Attention| First-Level | 5K| 5|[pipeline/icd9_cnn_att_workbook.ipynb](pipeline/icd9_cnn_att_workbook.ipynb) </br>  F1-score:72.8|
| Hierarchical LSTM Attention | First-level| 5k|5| [pipeline/icd9_hatt_workbook.ipynb](pipeline/icd9_hatt_workbook.ipynb)</br> This model was implemented based on [Yang et al. (2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) which specifically targets document classifications. It has two levels of attention mechanisms, the first one creates vectors that represent each sentence, using attention mechanism across words; and the second level creates a vector that represent the document using attention mechanisms across sentences. F1-score: 67.6|   


### Classification into most common ICD-9 Codes in the bottom of the ICD-9 Hierarchy (leaves)   
| Model | ICD 9 code level| N. Records | Epochs | Notebook |
| --- | --- | --- | --- | --- |
| Baseline | First-Level |46K and 5K| -|[baseline/mimic_icd9_baseline.ipynb](baseline/mimic_icd9_baseline.ipynb) <br/> - Some Initial Exploration with Python and Sql </br> - Basic Baseline Model: For the basic baseline, we make a fixed prediction corresponding to the top 4 ICD-9 codes for 46K records </br> - NN Baseline Model: A neural network (not Recurrent) with one hidden layer, with relu activation on the hidden layer and sigmoid activation on the output layer. Using cross entropy loss,which is the loss functions for multilabel classification (using Tensorflow), using 5K records. F1-score: 35  | 
| CNN for top 20 leaf icd-9 codes | Leaf | 46K | 7 | [icd9_cnn/cnn_top20_leave.ipynb](icd9_cnn/cnn_top20_leave.ipynb) </br> Classifies clinical notes into the 20 most common ICD-9 that are in the bottom of the ICD-9 hierarchy (leaves), this run was for comparison with previous work.  F1-score:72.4  |

### Classification into top level codes in the ICD-9 hierarchy with 52.6K records
| Model | ICD 9 code level| N. Records | Epochs | Notebook |
| --- | --- | --- | --- | --- |
| CNN | First-Level | 52.6K | - | [pipeline/icd9_cnn_50K_run.ipynb](/pipeline/icd9_cnn_50K_run.ipynb) </br> F1-score: 79.7 |
| CNN with Attention | First-Level | 52.6K | - | [pipeline/icd9_cnn_att_50K_records.ipynb](pipeline/icd9_cnn_att_50K_records.ipynb) </br> F1-score: 78.2.At this stage, the CNN ATT model still overfits: even though it had the highest score during the experimental runs with 5K records and 5 epochs, it didn’t reach the best f1-score when running it with the full data set. Further work would explore hyper-parameters tuning and evaluating the number of parameters to attempt undoing the over fitting situation.|



## Model Python modules

| Model | Python module |
| --- | --- |
| LSTM | [pipeline/lstm_model.py](pipeline/lstm_model.py) |
| CNN | [pipeline/icd9_cnn_model.py](pipeline/icd9_cnn_model.py)  |
| Attention Layer |[pipeline/attention_util.py](pipeline/attention_util.py)  |
| LSTM_ATT | [pipeline/icd9_lstm_att_model.py](pipeline/icd9_lstm_att_model.py)   |
| CNN_ATT | [pipeline/icd9_cnn_att.py](pipeline/icd9_cnn_att.py)   |
| Hierarchical LSTM Attention | [pipeline/hatt_model.py](pipeline/hatt_model.py)  |

## Helper classes for Preprocessing

| Helper | Python module |
| --- | --- |
| Filtering clinical-notes to keep the ones that have been assigned the top common N icd-9 codes (this is a multi-label),  removing any code from the label that is not in the top N | [pipeline/database_selection.py](pipeline/database_selection.py) |
| Three main methods: (1) Splits input file in training, valiation and test  (2) Replace leaf icd9-code with its grandparent in the first level  (3) Calculates and Diplay F1 scores for a set of possible thresholds| [pipeline/helpers.py](pipeline/helpers.py) |
| functions necessary to vectorize the ICD labels and text inputs 
(I didn't implement this module)| [pipeline/vectorization.py](pipeline/vectorization.py) |
