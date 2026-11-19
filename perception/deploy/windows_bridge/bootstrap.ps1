param($payload)
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator SSH required' }
$root = Join-Path $env:ProgramData 'ParadigmVisitorBridge'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$log = Join-Path $root 'bootstrap.log'
$mutex = New-Object Threading.Mutex($false, 'Global\ParadigmBridgeBootstrap')
if (!$mutex.WaitOne(0)) { throw 'Another installation is running' }
try {
    $taskName = 'Paradigm Visitor Bridge Watchdog'
    $oldTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    $oldTaskXml = if ($oldTask) { Export-ScheduledTask -TaskName $taskName } else { $null }
    $install = Join-Path $root 'app'
    if ($oldTask) {
        $working = $oldTask.Actions | Select-Object -First 1 -ExpandProperty WorkingDirectory
        if (!$working -or !(Test-Path (Join-Path $working 'run_bridge.py'))) { throw 'Existing task directory cannot be resolved' }
        $install = $working
    }
    New-Item -ItemType Directory -Force -Path $install | Out-Null
    $cfgPath = Join-Path $install 'config.json'
    if (Test-Path $cfgPath) {
        $cfg = Get-Content -LiteralPath $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($cfg.shared_token -cne $payload.token) { throw 'Existing token differs; use current token' }
        if ($cfg.port -ne 8765) { throw 'This installer requires bridge port 8765' }
    }
    $assets = Join-Path $root 'offline'
    New-Item -ItemType Directory -Force -Path $assets | Out-Null
    if (!$payload.offline) { throw 'Offline package missing; rebuild robot image' }
    foreach ($item in $payload.offline.PSObject.Properties) {
        if ($item.Name -notmatch '^[A-Za-z0-9_.-]+\.(exe|whl)$') { throw 'Invalid asset name' }
        $assetPath = Join-Path $assets $item.Name
        [IO.File]::WriteAllBytes($assetPath, [Convert]::FromBase64String($item.Value.content))
        if ((Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash -ne $item.Value.sha256) { throw 'Offline asset hash mismatch' }
    }
    Add-Content -LiteralPath $log -Value 'Offline package verified; preparing Python'
    $python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (!(Test-Path $python)) {
        $python = Join-Path $root 'Python312\python.exe'
        if (!(Test-Path $python)) {
            $exe = Join-Path $assets 'python-3.12.10-amd64.exe'
            $signature = Get-AuthenticodeSignature -LiteralPath $exe
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') { throw 'Python installer signature invalid' }
            $process = Start-Process -FilePath $exe -ArgumentList @('/quiet','InstallAllUsers=0',('TargetDir="' + (Join-Path $root 'Python312') + '"'),'Include_test=0','Include_launcher=0','Include_symbols=0','Include_debug=0','PrependPath=0') -Wait -PassThru -WindowStyle Hidden
            if ($process.ExitCode -notin @(0,3010)) { throw 'Python installation failed' }
        }
    }
    $venv = Join-Path $install '.venv'
    if (!(Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
        & $python -m venv $venv
        if ($LASTEXITCODE) { throw 'venv creation failed' }
    }
    $python = Join-Path $venv 'Scripts\python.exe'
    Add-Content -LiteralPath $log -Value 'Installing dependencies from local wheels (no internet download)'
    & $python -m pip --isolated install --no-index --no-cache-dir --disable-pip-version-check --find-links $assets 'playwright==1.51.0' 'greenlet==3.1.1' 'pyee==12.1.1' 'typing_extensions==4.12.2' >> $log 2>&1
    if ($LASTEXITCODE) { throw 'Dependency installation failed' }
    $edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") | Where-Object { Test-Path $_ }
    if (!$edge) { throw 'Install Microsoft Edge before initialization' }
    $backup = Join-Path $install ('backup-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $backup | Out-Null
    $allowed = @('bridge_core.py','http_server.py','edge_backend.py','run_bridge.py','login_edge.py','bridge_watchdog.ps1','remote_start_bridge.ps1','config.example.json')
    foreach ($name in $allowed) {
        if (!$payload.files.$name) { throw 'Incomplete package' }
        if (Test-Path (Join-Path $install $name)) { Copy-Item -LiteralPath (Join-Path $install $name) -Destination $backup }
    }
    $stage = Join-Path $backup 'staged'
    New-Item -ItemType Directory -Path $stage | Out-Null
    foreach ($name in $allowed) { [IO.File]::WriteAllBytes((Join-Path $stage $name), [Convert]::FromBase64String($payload.files.$name)) }
    & $python -m py_compile (Join-Path $stage 'bridge_core.py') (Join-Path $stage 'http_server.py') (Join-Path $stage 'edge_backend.py') (Join-Path $stage 'run_bridge.py')
    if ($LASTEXITCODE) { throw 'Package syntax validation failed' }
    if ($oldTask) { Stop-ScheduledTask -TaskName $taskName }
    Stop-ScheduledTask -TaskName 'Paradigm Visitor Bridge' -ErrorAction SilentlyContinue
    try {
        Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)"
            if ($proc.CommandLine -notlike "*$install*run_bridge.py*") { throw 'Port 8765 occupied by another program' }
            Stop-Process -Id $_.OwningProcess -Force
        }
        foreach ($name in $allowed) { [IO.File]::WriteAllBytes((Join-Path $install $name), [Convert]::FromBase64String($payload.files.$name)) }
        if (!(Test-Path $cfgPath)) {
            $cfg = Get-Content (Join-Path $install 'config.example.json') -Raw -Encoding UTF8 | ConvertFrom-Json
            $cfg.shared_token = $payload.token
            $cfg.edge.printer_name = ''
            $cfg | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $cfgPath -Encoding UTF8
        }
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + (Join-Path $install 'bridge_watchdog.ps1') + '"') -WorkingDirectory $install
        $taskUser = if ($oldTask) { $oldTask.Principal.UserId } else { $identity.Name }
        $taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Highest
        $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $taskUser) -Principal $taskPrincipal -Settings $settings -Force | Out-Null
        $robotAddress = ($env:SSH_CLIENT -split ' ')[0]
        $parsedAddress = $null
        if (![Net.IPAddress]::TryParse($robotAddress, [ref]$parsedAddress)) { throw 'SSH client address unavailable' }
        $ruleName = 'ParadigmBridge-' + ($robotAddress -replace ':','-')
        if (!(Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue)) { New-NetFirewallRule -Name $ruleName -DisplayName 'Visitor bridge robot' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 -RemoteAddress $robotAddress | Out-Null }
        Start-ScheduledTask -TaskName $taskName
        $ready = $false
        for ($i=0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            try { if ((Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 2).state -eq 'running') { $ready=$true; break } } catch {}
        }
        if (!$ready) { throw 'Bridge not ready; Windows desktop user must be logged in' }
    } catch {
        if ($oldTask) {
            Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
            Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
                $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)"
                if ($proc.CommandLine -like "*$install*run_bridge.py*") { Stop-Process -Id $_.OwningProcess -Force }
            }
            Get-ChildItem -LiteralPath $backup -File | Copy-Item -Destination $install -Force
            Register-ScheduledTask -TaskName $taskName -Xml $oldTaskXml -Force | Out-Null
            Start-ScheduledTask -TaskName $taskName
        }
        throw
    }
    Write-Output 'BRIDGE_BOOTSTRAP_OK'
} catch {
    Add-Content -LiteralPath $log -Value ($_.Exception.Message)
    Write-Error 'Bootstrap failed; see bootstrap.log'
    exit 1
} finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
