-- filter out patients who don’t have at least one of the top  icd9-codes 
-- and only include these icd9-codes


select  * 
from diagnoses_icd 
where icd9_code IN ( select icd9_code from (select icd9_code, COUNT(DISTINCT SUBJECT_ID) subjects_qty 
from diagnoses_icd 
where SUBSTRING(icd9_code from 1 for 1) != 'V' 
group by icd9_code 
order by subjects_qty  
desc  limit 20) as icd9_subject_list)
order by SUBJECT_ID, HADM_ID ;


