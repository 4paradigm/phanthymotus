$ErrorActionPreference = "Stop"

$taskName = "Paradigm Visitor Bridge Watchdog"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Error "Bridge watchdog task is not installed."
    exit 2
}

Start-ScheduledTask -TaskName $taskName
Write-Output "STARTED"
