$LogFile = "D:\Issac\polyarb_lab\data\reports\senate_mutex_probe_20260322.log"
cd D:\Issac\polyarb_lab
py -3 D:\Issac\polyarb_lab\_probe_senate_mutex.py 2>&1 | Tee-Object -FilePath $LogFile
