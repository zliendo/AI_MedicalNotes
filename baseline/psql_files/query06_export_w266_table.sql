select HADM_ID, SUBJECT_ID, CHARTDATE, regexp_replace(TEXT, E'[\\n\\r]+', ' ', 'g' ),  ICD9_CODES
from W266_DISCHARGE_NOTE_ICD9_CODES;
