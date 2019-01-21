--select hadm_id, max(subject_id),  string_agg(icd9_code, ',') from diagnoses_icd where hadm_id = '145834' group by hadm_id

-- aggregates icd9-codes in one row
-- generates diagnoses_icd_codes.csv

select hadm_id, max(subject_id) subject_id,  string_agg(icd9_code, ' ') icd9_codes 
from diagnoses_icd 
where icd9_code IN ( select icd9_code from (select icd9_code, COUNT(DISTINCT SUBJECT_ID) subjects_qty 
from diagnoses_icd 
where SUBSTRING(icd9_code from 1 for 1) != 'V' 
group by icd9_code 
order by subjects_qty  
desc  limit 20) as icd9_subject_list)
group by hadm_id;