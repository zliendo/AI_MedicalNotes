-- We identify the top 20 labels based on number of patients with that label.

select icd9_code, COUNT(DISTINCT SUBJECT_ID) subjects_qty 
from diagnoses_icd 
where SUBSTRING(icd9_code from 1 for 1) != 'V' 
group by icd9_code 
order by subjects_qty  
desc  limit 20;


