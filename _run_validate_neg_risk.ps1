$LogFile = "D:\Issac\polyarb_lab\data\reports\neg_risk_fix_validation_20260322.log"
cd D:\Issac\polyarb_lab
py -3 D:\Issac\polyarb_lab\_validate_neg_risk_fix.py 2>&1 | Tee-Object -FilePath $LogFile
