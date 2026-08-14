#!/bin/bash

# Exit immediately if any command fails
set -e

# Configuration (Adjust these to match your setup)
BRANCH="main"
GUNICORN_SERVICE="gunicorn"  # Change to your actual systemd service name (e.g., myapp.service)
VENV_DIR="venv"

echo "=========================================="
echo "🚀 Starting Automated Deployment..."
echo "=========================================="

# 1. Pull latest changes from Git
echo "📥 Pulling updates from branch: $BRANCH..."
git pull origin $BRANCH

# 2. Update Python dependencies if virtualenv exists
if [ -d "$VENV_DIR" ]; then
    echo "📦 Activating virtual environment and updating packages..."
    source $VENV_DIR/bin/activate
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
else
    echo "⚠️  No virtual environment found at ./$VENV_DIR. Skipping pip update."
fi

# 3. Restart Gunicorn app backend
echo "🔄 Restarting Gunicorn service ($GUNICORN_SERVICE)..."
sudo systemctl restart $GUNICORN_SERVICE

# 4. Reload Nginx reverse proxy
echo "🌐 Reloading Nginx..."
sudo systemctl reload nginx

echo "=========================================="
echo "✅ Deployment finished successfully!"
echo "=========================================="

# 5. Display brief service status check
sudo systemctl status $GUNICORN_SERVICE --no-pager -n 5
