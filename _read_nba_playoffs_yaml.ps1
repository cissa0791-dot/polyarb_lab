$yaml = Get-Content 'D:\Issac\polyarb_lab\data\reports\discovered_constraints_nba_playoffs_20260322.yaml' -Raw
$lines = $yaml -split "`n"
$results = @()
$in_block = $false
$block_lines = @()
for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match 'nba_finals_implies_playoffs') {
        $in_block = $true
        $start = [Math]::Max(0, $i - 10)
        $end = [Math]::Min($lines.Length - 1, $i + 18)
        $results += $lines[$start..$end]
        $results += '---'
    }
}
$results -join "`n"
