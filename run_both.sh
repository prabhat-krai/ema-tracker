#!/usr/bin/env bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo "======================================"
echo "    Running INDIA Market Screener"
echo "======================================"
python3 -m src.main

echo ""
echo "======================================"
echo "      Running USA Market Screener"
echo "======================================"
python3 -m src.main --usa

echo ""
echo "Done! Both markets have been scanned and logs are updated."
