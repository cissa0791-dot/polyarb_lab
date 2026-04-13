$LogFile = "D:\Issac\polyarb_lab\data\reports\nba_playoffs_discovery_20260322.log"
$YamlOut = "D:\Issac\polyarb_lab\data\reports\discovered_constraints_nba_playoffs_20260322.yaml"

py -3 D:\Issac\polyarb_lab\scripts\discover_cross_market_constraints.py `
    --settings D:\Issac\polyarb_lab\config\settings.yaml `
    --market-limit 3000 `
    --rank-current-execution `
    --max-constraints 100 `
    --out-path $YamlOut `
    2>&1 | Tee-Object -FilePath $LogFile
