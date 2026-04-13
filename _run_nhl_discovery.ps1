$LogFile = "D:\Issac\polyarb_lab\data\reports\nhl_discovery_20260322.log"
$YamlOut = "D:\Issac\polyarb_lab\data\reports\discovered_constraints_nhl_20260322.yaml"

py -3 D:\Issac\polyarb_lab\scripts\discover_cross_market_constraints.py `
    --settings D:\Issac\polyarb_lab\config\settings.yaml `
    --market-limit 3000 `
    --rank-current-execution `
    --max-constraints 50 `
    --out-path $YamlOut `
    2>&1 | Tee-Object -FilePath $LogFile
