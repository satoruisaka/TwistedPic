#!/bin/bash
# Script to execute port forwarding and start TwistedPic server

echo "Preparing to start TwistedPic server..."

echo "1. Checking Ollama status..."

# Check if Ollama is running (using curl from WSL)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running. Please start Ollama first."
    echo "   Run: ollama serve"
    exit 1
fi
echo "✅ Ollama is running"

echo "2. Checking TwistedPair status..."

# Check if TwistedPair is running (using curl from WSL)
if ! curl -s http://localhost:8001/api/tags > /dev/null 2>&1; then
    echo "❌ TwistedPair is not running. Please start TwistedPair first."
    echo "   Run: ./startMRA2_twistedpair.sh"
    exit 1
fi
echo "✅ TwistedPair is running"

echo "3. Configuring port forwarding..."

# Configure port forwarding on Windows side

powershell.exe -Command "
  \$wslIP = (wsl hostname -I).Trim();
  netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0 2>\$null;
  netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=0.0.0.0 2>\$null;
  netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=\$wslIP;
  netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=\$wslIP;
  Write-Host 'Port forwarding active for 8000 and 5000 on WSL IP:' \$wslIP -ForegroundColor Green
"

# Check if PowerShell command succeeded
if [ $? -ne 0 ]; then
    echo "Port forwarding setup failed. Exiting."
    exit 1
fi

echo "Port forwarding setup complete."

# Activate venv and start server
echo "4. Starting virtual env in TwistedPic directory..."
cd ~/../../mnt/c/Users/sator/linuxproject/TwistedPic || { echo "Failed to change directory to TwistedPic"; exit 1; }
source ./.venv/bin/activate

echo "5. Starting TwistedPic server..."
python server.py

