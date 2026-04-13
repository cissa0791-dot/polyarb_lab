# _run_reward_maker_probe.ps1
# reward_aware_single_market_maker_probe — PowerShell runner
# Run from repo root: .\\_run_reward_maker_probe.ps1

param(
    [switch]$Verbose,
    [string]$LogLevel = "INFO",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$args_list = @(
    "research_lines/reward_aware_maker_probe/run_maker_probe.py",
    "--log-level", $LogLevel
)

if ($Verbose) { $args_list += "--verbose" }
if ($OutputDir -ne "") { $args_list += @("--output-dir", $OutputDir) }

Write-Host "reward_aware_maker_probe: starting probe..." -ForegroundColor Cyan
py -3 @args_list
