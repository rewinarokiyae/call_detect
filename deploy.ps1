
param(
    [Parameter(Mandatory=$true)]
    [string]$SSHCommand
)

# 1. Parse SSH Details
# Expected format: ssh -p <PORT> root@<IP>
if ($SSHCommand -match "ssh -p (\d+) (.*)@(.*)") {
    $Port = $matches[1]
    $User = $matches[2]
    $IP = $matches[3]
} elseif ($SSHCommand -match "ssh (.*)@(.*)") {
    # Default port 22 logic if no -p
    $Port = "22"
    $User = $matches[1]
    $IP = $matches[2]
} else {
    Write-Host "Error: Could not parse SSH command. Format should be: ssh -p PORT user@IP" -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/4] Targeted Remote: $User@$IP Port $Port" -ForegroundColor Cyan

# 2. Package Code
Write-Host "[2/4] Packaging 'src_asvspoof' and requirements..." -ForegroundColor Cyan
$ZipFile = "payload.zip"
if (Test-Path $ZipFile) { Remove-Item $ZipFile }

# Compress src_asvspoof and requirements.txt
Compress-Archive -Path "src_asvspoof", "requirements.txt" -DestinationPath $ZipFile -Force

# 3. Upload Payload
Write-Host "[3/4] Uploading payload to Vast.ai (Enter password if asked)..." -ForegroundColor Cyan
# scp -P <PORT> payload.zip <USER>@<IP>:/workspace/
$ScpCmd = "scp -P $Port $ZipFile ${User}@${IP}:/workspace/"
Invoke-Expression $ScpCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed." -ForegroundColor Red
    exit 1
}

# 4. Remote Execution
Write-Host "[4/4] Executing Setup & Training on Remote GPU..." -ForegroundColor Cyan

$RemoteScript = @"
cd /workspace
# Clean previous
rm -rf src_asvspoof requirements.txt
# Unzip
unzip -o payload.zip
# Setup
bash src_asvspoof/setup_vast_ai.sh
# Run Rigorous Training
python -m src_asvspoof.train_rigorous
"@

# SSH Command to execute the block above
# We define it carefully to handle newlines
$SshExec = "ssh -p $Port -t ${User}@${IP} ""$RemoteScript"""
Invoke-Expression $SshExec

Write-Host "`nDeployment & Training Triggered!" -ForegroundColor Green
