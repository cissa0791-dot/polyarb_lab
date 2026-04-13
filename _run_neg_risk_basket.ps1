$LogFile = "D:\Issac\polyarb_lab\data\reports\neg_risk_basket_probe_20260322.log"
cd D:\Issac\polyarb_lab
py -3 D:\Issac\polyarb_lab\_probe_neg_risk_basket.py 2>&1 | Tee-Object -FilePath $LogFile
