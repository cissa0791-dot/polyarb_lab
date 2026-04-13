$yaml = Get-Content 'D:\Issac\polyarb_lab\data\reports\discovered_constraints_nhl_20260322.yaml' -Raw
$lines = $yaml -split "`n"
$results = @()
for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match 'nhl_champion_implies_playoffs') {
        $start = [Math]::Max(0, $i - 10)
        $end = [Math]::Min($lines.Length - 1, $i + 15)
        $results += $lines[$start..$end]
        $results += '---'
    }
}
$results -join "`n"
