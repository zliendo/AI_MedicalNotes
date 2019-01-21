

## ICD9-codes

The DIAGNOSES_ICD table contains the ICD9-codes assigned to a hospital admission. There could be many ICD9-codes assigned to one admission.
There are 57,786 admissions in this table, with a total of 651,047 ICD9-codes assigned.

Related previous research didn't work with all ICD9-codes but only with the most used in diagnoses reports[2] [3]. We will not consider ICD9-codes that start with "E" (additional information indicating the cause of injury or adverse event) nor "V" (codes used when the visit is due to circumstances other than disease or injury, e.g.: new born to indicate birth status)   

*	We identify the top 20 labels based on number of patients with that label. 
*	We then remove all patients who don’t have at least one of these labels,  
*	and then filter the set of labels for each patient to only include these labels.   

As a result, we get 45,293 admissions with 152,299 icd9-codes (only including the ones in the top 20)

Here is the list of the top 20 ICD9-codes that will be used in the baseline

```
select icd9_code, COUNT(DISTINCT SUBJECT_ID) subjects_qty 
from diagnoses_icd  where SUBSTRING(icd9_code from 1 for 1) != 'V' 
group by icd9_code order by subjects_qty  
desc  limit 20;

 icd9_code | subjects_qty
-----------+--------
 4019      |  17613
 41401     |  10775
 42731     |  10271
 4280      |   9843
 5849      |   7687
 2724      |   7465
 25000     |   7370
 51881     |   6719
 5990      |   5779
 2720      |   5335
 53081     |   5272
 2859      |   4993
 486       |   4423
 2851      |   4241
 2762      |   4177
 2449      |   3819
 496       |   3592
 99592     |   3560
 0389      |   3433
 5070      |   3396
(20 rows)


```

The file containing the list of admissions resulting of the filtering above is in this file: baseline\psql_files\diagnoses_icd_codes.csv
(note: we removed all files containing MIMIC data because they need authorization by MIMIC to access)

Here is the sql that created that file

```
select hadm_id, max(subject_id) subject_id,  string_agg(icd9_code, ' ') icd9_codes 
from diagnoses_icd 
where icd9_code IN ( select icd9_code from (select icd9_code, COUNT(DISTINCT SUBJECT_ID) subjects_qty 
from diagnoses_icd where SUBSTRING(icd9_code from 1 for 1) != 'V' 
group by icd9_code order by subjects_qty  desc  limit 20) as icd9_subject_list)
group by hadm_id;
```

(note: for the final model, do we consider the code's description and/or hierarchy?)

## Clinical Notes

The clinical notes related to an admission are located in the NOTEEVENTS and ADMISSION tables.
The ADMISSION table has only one type of clinical note, the one related to the preliminary diagnoses done during admission.
The NOTEEVENTS contains all the other type of clinical notes, here is a list of these types
```
mimic=# select category from noteevents group by category;
     category
-------------------
 ECG
 Respiratory
 Discharge summary
 Radiology
 Rehab Services
 Nursing/other
 Nutrition
 Pharmacy
 Social Work
 Case Management
 Physician
 General
 Nursing
 Echo
 Consult
(15 rows)

``` 
The baseline will  ONLY  use the 'Discharge Summary' clinical notes. (note: we may use the other clinical notes for the final project)
An example of one discharge summary note can be found at: baseline/psql_files/discharge_note_sample.out
(we removde all data related files since they need granted authorization by MIMIC)


It looks like the discharge summary can have Addendum, we will not include Addendums for the baseline
```
mimic=# select category, description  from noteevents  where category = 'Discharge summary' group by category, description;
     category      | description
-------------------+-------------
 Discharge summary | Report
 Discharge summary | Addendum
(2 rows)
```

It looks like there are duplicates entries for some admissions, for example, 
```
mimic=# select HADM_ID, SUBJECT_ID, CHARTDATE, CGID, ISERROR, substring(TEXT from 1 for 20)
mimic-# from noteevents
mimic-# where HADM_ID = '178053' and noteevents.category = 'Discharge summary'   and noteevents.DESCRIPTION = 'Report';

 hadm_id | subject_id |      chartdate      | cgid | iserror |      substring
---------+------------+---------------------+------+---------+----------------------
  178053 |      18976 | 2120-11-28 00:00:00 |      |         | Admission Date:  [**
  178053 |      18976 | 2120-11-28 00:00:00 |      |         | Admission Date:  [**
  178053 |      18976 | 2120-12-16 00:00:00 |      |         | Admission Date:  [**
  178053 |      18976 | 2120-12-16 00:00:00 |      |         | Admission Date:  [**
  178053 |      18976 | 2120-11-26 00:00:00 |      |         | Admission Date:  [**

(5 rows)
```
in this case the clinical notes are different, seems the did multiple entries for the same discharge, and the discharge happened in two different dates with the same admission_id (that looks like a mistake since a patient returning should get a new admission_id)

We will handle this situation for the final model.

##  Joining information from the clinical notes and the corresponding ICD9 codes

This is the sql statement that created a table with a join from the discharge summary notes and its ICD_CODES that are in the top 20.
```
DROP TABLE IF EXISTS W266_DISCHARGE_NOTE_ICD9_CODES;
CREATE TABLE  W266_DISCHARGE_NOTE_ICD9_CODES  AS 

select noteevents.HADM_ID, dianoses_top20_icd.SUBJECT_ID, noteevents.CHARTDATE, noteevents.TEXT,  dianoses_top20_icd.ICD9_CODES
from noteevents 
JOIN
(select HADM_ID, max(SUBJECT_ID) SUBJECT_ID,  string_agg(ICD9_CODE, ' ') ICD9_CODES 
from diagnoses_icd 
where ICD9_CODE IN ( select ICD9_CODE from (select ICD9_CODE, COUNT(DISTINCT SUBJECT_ID) subjects_qty 
from diagnoses_icd 
where SUBSTRING(ICD9_CODE from 1 for 1) != 'V' 
group by ICD9_CODE  order by subjects_qty  
desc  limit 20) as icd9_subject_list)
group by HADM_ID ) as dianoses_top20_icd
ON (noteevents.HADM_ID = dianoses_top20_icd.HADM_ID)
where noteevents.category = 'Discharge summary'  and noteevents.DESCRIPTION = 'Report';


CREATE INDEX W266_DISCHARGE_NOTE_ICD9_CODES_index 
ON W266_DISCHARGE_NOTE_ICD9_CODES(HADM_ID) ;


```

the file is about 474 MB

## Representing clinical notes for the baseline

Previous research represents this documents as bag-of-words vectors [1]. In particular, it takes the 10,000 tokens with the largest tf-idf scores from the training.   

(note for the final model: we could use here POS tagging, parsing and entity recognition)

## Basic Baseline Model
For the basic baseline, we make a fixed prediction corresponding to the top 4 ICD-9 codes for all records

## NN Baseline Model
A neural network (not Recurrent) with one hidden layer, with relu activation on the hidden layer and sigmoid activation on the output layer.   Using cross entropy loss,which is the loss functions for multilabel classification [4]   
   

## Evaluation
rank loss metric to evaluate performance [3] and F1 score [2] 

## References
[1] Diagnosis code assignment: models and evaluation metrics. Journal of the American Medical Informatics   
[2] Applying Deep Learning to ICD-9 Multi-label Classification from Medical Records   
[3] ICD-9 Coding of Discharge Summaries   
[4] Large-scale Multi-label Text Classification - Revisiting Neural Networks   
