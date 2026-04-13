$LogFile = "D:\Issac\polyarb_lab\data\reports\cl_progression_probe_20260322.log"
py -3 D:\Issac\polyarb_lab\_probe_cl_progression.py 2>&1 | Tee-Object -FilePath $LogFile
