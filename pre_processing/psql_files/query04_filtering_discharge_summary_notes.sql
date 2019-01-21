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


