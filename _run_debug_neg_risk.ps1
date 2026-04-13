$LogFile = "D:\Issac\polyarb_lab\data\reports\neg_risk_debug_20260322.log"
cd D:\Issac\polyarb_lab
py -3 D:\Issac\polyarb_lab\_debug_neg_risk_edge.py 2>&1 | Tee-Object -FilePath $LogFile
