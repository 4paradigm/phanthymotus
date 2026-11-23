$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator SSH required' }

$upload = 'D:\ParadigmVisitorAutomation\upload'
$corePath = Join-Path $upload 'core.json'
$manifestPath = Join-Path $upload 'manifest.json'
if (!(Test-Path $corePath) -or !(Test-Path $manifestPath)) { throw 'Deployment files missing' }
$core = Get-Content -LiteralPath $corePath -Raw -Encoding UTF8 | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$root = 'D:\ParadigmVisitorAutomation'
$install = Join-Path $root 'app'
$assets = Join-Path $root 'offline'
$log = Join-Path $root 'install.log'
New-Item -ItemType Directory -Force -Path $install,$assets | Out-Null

$mutex = New-Object Threading.Mutex($false, 'Global\ParadigmVisitorAutomationInstall')
if (!$mutex.WaitOne(0)) { throw 'Another installation is running' }
try {
    foreach ($fileInfo in $manifest.files) {
        if ($fileInfo.name -notmatch '^[A-Za-z0-9_.-]+\.(exe|whl)$') { throw 'Invalid asset name' }
        $assetPath = Join-Path $assets $fileInfo.name
        $stream = [IO.FileStream]::new($assetPath, [IO.FileMode]::Create)
        try {
            for ($i = 0; $i -lt $fileInfo.chunk_count; $i++) {
                $chunk = Join-Path $upload ("{0}.{1:D4}.chunk" -f $fileInfo.name, $i)
                if (!(Test-Path $chunk)) { throw "Missing chunk: $chunk" }
                $bytes = [IO.File]::ReadAllBytes($chunk)
                $stream.Write($bytes, 0, $bytes.Length)
            }
        } finally { $stream.Dispose() }
        if ((Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash -ne $fileInfo.sha256) {
            throw 'Offline asset hash mismatch'
        }
    }

    $python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (!(Test-Path $python)) {
        $python = Join-Path $root 'Python312\python.exe'
        if (!(Test-Path $python)) {
            $installer = Join-Path $assets 'python-3.12.10-amd64.exe'
            $signature = Get-AuthenticodeSignature -LiteralPath $installer
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
                throw 'Python installer signature invalid'
            }
            $process = Start-Process -FilePath $installer -ArgumentList @(
                '/quiet','InstallAllUsers=0',('TargetDir="' + (Join-Path $root 'Python312') + '"'),
                'Include_test=0','Include_launcher=0','Include_symbols=0','Include_debug=0','PrependPath=0'
            ) -Wait -PassThru -WindowStyle Hidden
            if ($process.ExitCode -notin @(0,3010)) { throw 'Python installation failed' }
        }
    }
    $venv = Join-Path $install '.venv'
    if (!(Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
        & $python -m venv $venv
        if ($LASTEXITCODE) { throw 'venv creation failed' }
    }
    $python = Join-Path $venv 'Scripts\python.exe'
    & $python -m pip --isolated install --no-index --no-cache-dir --disable-pip-version-check `
        --find-links $assets 'playwright==1.51.0' 'greenlet==3.1.1' 'pyee==12.1.1' 'typing_extensions==4.12.2' >> $log 2>&1
    if ($LASTEXITCODE) { throw 'Offline dependency installation failed' }

    $allowed = @('bridge_core.py','edge_backend.py','ssh_worker.py','login_edge.py','config.example.json','README.txt')
    foreach ($name in $allowed) {
        if (!$core.core_files.$name) { throw "Incomplete package: $name" }
        [IO.File]::WriteAllBytes((Join-Path $install $name), [Convert]::FromBase64String($core.core_files.$name))
    }
    & $python -m py_compile (Join-Path $install 'bridge_core.py') (Join-Path $install 'edge_backend.py') `
        (Join-Path $install 'ssh_worker.py') (Join-Path $install 'login_edge.py')
    if ($LASTEXITCODE) { throw 'Automation syntax validation failed' }

    $cfgPath = Join-Path $install 'config.json'
    if (!(Test-Path $cfgPath)) {
        $cfg = Get-Content (Join-Path $install 'config.example.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $cfg.PSObject.Properties.Remove('bind_host')
        $cfg.PSObject.Properties.Remove('port')
        $cfg.PSObject.Properties.Remove('shared_token')
        $cfg.database_path = 'data/automation.db'
        $cfg.edge.printer_name = ''
        $cfg | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $cfgPath -Encoding UTF8
    }

    # Updating an existing install must release the old persistent Edge
    # profile before the new login task starts. Prefer the cooperative stop
    # signal; only close Edge processes whose command line names this profile.
    $cfg = Get-Content -LiteralPath $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $loginTask = 'Paradigm Visitor Login'
    $loginState = Get-ScheduledTask -TaskName $loginTask -ErrorAction SilentlyContinue
    if ($loginState -and $loginState.State -eq 'Running') {
        $dataDir = Join-Path $install 'data'
        New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
        Set-Content -LiteralPath (Join-Path $dataDir 'login.stop') -Value 'upgrade' -Encoding ascii
        for ($wait = 0; $wait -lt 40; $wait++) {
            $loginState = Get-ScheduledTask -TaskName $loginTask -ErrorAction SilentlyContinue
            if (!$loginState -or $loginState.State -ne 'Running') { break }
            Start-Sleep -Milliseconds 250
        }
        if ($loginState -and $loginState.State -eq 'Running') {
            Stop-ScheduledTask -TaskName $loginTask -ErrorAction SilentlyContinue
            for ($wait = 0; $wait -lt 40; $wait++) {
                $loginState = Get-ScheduledTask -TaskName $loginTask -ErrorAction SilentlyContinue
                if (!$loginState -or $loginState.State -ne 'Running') { break }
                Start-Sleep -Milliseconds 250
            }
        }
    }
    $profilePath = [string]$cfg.edge.profile_dir
    if (!$profilePath) { $profilePath = 'data/edge-profile' }
    if (![IO.Path]::IsPathRooted($profilePath)) { $profilePath = Join-Path $install $profilePath }
    $profilePath = [IO.Path]::GetFullPath($profilePath)
    $edgeProcesses = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'msedge.exe' -and $_.CommandLine -and
        $_.CommandLine.IndexOf($profilePath, [StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    foreach ($edgeProcess in $edgeProcesses) {
        $window = Get-Process -Id $edgeProcess.ProcessId -ErrorAction SilentlyContinue
        if ($window) { $null = $window.CloseMainWindow() }
    }
    if ($edgeProcesses.Count) {
        Start-Sleep -Seconds 3
        $edgeProcesses = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'msedge.exe' -and $_.CommandLine -and
            $_.CommandLine.IndexOf($profilePath, [StringComparison]::OrdinalIgnoreCase) -ge 0
        })
        foreach ($edgeProcess in $edgeProcesses) {
            Stop-Process -Id $edgeProcess.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    for ($wait = 0; $wait -lt 40; $wait++) {
        $loginState = Get-ScheduledTask -TaskName $loginTask -ErrorAction SilentlyContinue
        if (!$loginState -or $loginState.State -ne 'Running') { break }
        Start-Sleep -Milliseconds 250
    }
    if ($loginState -and $loginState.State -eq 'Running') {
        throw 'Cannot release the previous visitor Edge task. Close its Edge window and retry.'
    }
    $loginStopPath = $install + '\data\login.stop'
    Remove-Item -LiteralPath $loginStopPath -Force -ErrorAction SilentlyContinue

    # Remove the obsolete network bridge completely.
    Stop-ScheduledTask -TaskName 'Paradigm Visitor Bridge Watchdog' -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName 'Paradigm Visitor Bridge Watchdog' -Confirm:$false -ErrorAction SilentlyContinue
    Stop-ScheduledTask -TaskName 'Paradigm Visitor Bridge' -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName 'Paradigm Visitor Bridge' -Confirm:$false -ErrorAction SilentlyContinue
    Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)"
        if ($process.CommandLine -like '*ParadigmVisitorBridge*') { Stop-Process -Id $_.OwningProcess -Force }
    }
    Get-NetFirewallRule -Name 'ParadigmBridge-*' -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue

    $taskName = 'Paradigm Visitor Automation'
    # Interactive desktop access is required for Edge, but the Python
    # console is not: discovery polls must not repeatedly steal focus.
    $pythonWindowless = Join-Path $venv 'Scripts\pythonw.exe'
    if (!(Test-Path -LiteralPath $pythonWindowless)) { throw 'Windowless Python executable missing' }
    $action = New-ScheduledTaskAction -Execute $pythonWindowless `
        -Argument ('"' + (Join-Path $install 'ssh_worker.py') + '" --config "' + $cfgPath + '"') `
        -WorkingDirectory $install
    $taskPrincipal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Highest
    # -StartWhenAvailable:  if Start-ScheduledTask is called while the
    #   interactive session is unavailable (locked / disconnected / user
    #   switching), the task starts as soon as the session comes back
    #   instead of silently dropping the start request.
    # -MultipleInstances Queue: if a start request arrives while the
    #   previous short-lived worker is still exiting (or still processing),
    #   the new run is queued rather than silently discarded.  The worker
    #   drains all *.request.json files on each run, so a queued instance
    #   either picks up late arrivals (good) or exits after the 10 s idle
    #   grace period (harmless).
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances Queue -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
        -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $taskPrincipal -Settings $settings -Force | Out-Null

    $loginAction = New-ScheduledTaskAction -Execute $pythonWindowless `
        -Argument ('"' + (Join-Path $install 'login_edge.py') + '" --automatic --config "' + $cfgPath + '"') `
        -WorkingDirectory $install
    $loginSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $loginTask -Action $loginAction -Principal $taskPrincipal -Settings $loginSettings -Force | Out-Null
    Start-ScheduledTask -TaskName $loginTask

    New-Item -ItemType Directory -Force -Path (Join-Path $install 'data\jobs') | Out-Null
    Remove-Item -Recurse -Force $upload -ErrorAction SilentlyContinue
    Write-Output 'SSH_AUTOMATION_OK'
} catch {
    Add-Content -LiteralPath $log -Value $_.Exception.Message
    Write-Error 'SSH automation installation failed; see install.log'
    exit 1
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
