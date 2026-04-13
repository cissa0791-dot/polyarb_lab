$LogFile = "D:\Issac\polyarb_lab\data\reports\slug_probe_20260322.log"
py -3 D:\Issac\polyarb_lab\_probe_slug_patterns.py 2>&1 | Tee-Object -FilePath $LogFile
