# Shared by the SSH preflight and bootstrap so both select the same installation.
function Get-BridgeInstallDirectory {
    $task = Get-ScheduledTask -TaskName 'Paradigm Visitor Bridge Watchdog' -ErrorAction SilentlyContinue
    if ($task) {
        $working = $task.Actions | Select-Object -First 1 -ExpandProperty WorkingDirectory
        if (!$working -or !(Test-Path -LiteralPath (Join-Path $working 'run_bridge.py'))) {
            throw 'Existing task directory cannot be resolved'
        }
        return $working
    }
    $candidates = @('D:\ParadigmVisitorBridge\app', 'D:\ParadigmVisitorBridge',
                    'C:\ProgramData\ParadigmVisitorBridge\app', 'C:\ProgramData\ParadigmVisitorBridge')
    $existing = @($candidates | Where-Object { Test-Path -LiteralPath (Join-Path $_ 'config.json') })
    if ($existing.Count -gt 1) { throw 'Multiple bridge configurations found; resolve the active installation first' }
    if ($existing.Count -eq 1) { return $existing[0] }
    return 'D:\ParadigmVisitorBridge\app'
}

function Get-BridgeTokenInfo {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator SSH required' }
    $cfgPath = Join-Path (Get-BridgeInstallDirectory) 'config.json'
    if (!(Test-Path -LiteralPath $cfgPath)) { return @{ installed = $false } }
    # Do not emit config content if parsing fails: it contains credentials.
    try { $cfg = Get-Content -LiteralPath $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { throw 'Existing bridge configuration is unreadable' }
    if ($cfg.port -ne 8765) { throw 'This installer requires bridge port 8765' }
    if (!$cfg.shared_token -or $cfg.shared_token.Length -lt 24) { throw 'Existing bridge token is invalid' }
    return @{ installed = $true; token = [string]$cfg.shared_token }
}
