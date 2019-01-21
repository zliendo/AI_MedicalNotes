DROP TABLE IF EXISTS W266_DIAGNOSES_TOP_ICD9_CODES;
CREATE TABLE  W266_DIAGNOSES_TOP_ICD9_CODES  AS 

select hadm_id, max(subject_id) subject_id,  string_agg(icd9_code, ' ') icd9_codes 
from diagnoses_icd 
where icd9_code IN ( select icd9_code from (select icd9_code, COUNT(DISTINCT SUBJECT_ID) subjects_qty 
from diagnoses_icd 
where SUBSTRING(icd9_code from 1 for 1) != 'V' 
group by icd9_code 
order by subjects_qty  
desc  limit 20) as icd9_subject_list)
group by hadm_id;

CREATE INDEX W266_DIAGNOSES_TOP_ICD9_CODES_index 
ON W266_DIAGNOSES_TOP_ICD9_CODES (HADM_ID) ;