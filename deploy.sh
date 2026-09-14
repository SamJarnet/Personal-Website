#!/bin/bash
set -e

cd /home/samuelpi/git/Personal-Website

echo "Pulling latest changes..."
git pull origin main

if [ -f requirements.txt ]; then
    echo "Updating dependencies..."
    venv/bin/pip install -r requirements.txt
fi

echo "Restarting website..."
sudo systemctl restart gunicorn
sudo systemctl reload nginx

echo "Done."
sudo systemctl is-active gunicorn
sudo systemctl is-active nginx
