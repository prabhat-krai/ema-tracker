#!/usr/bin/env bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Activate virtual environment and run the script for the India market
source venv/bin/activate
python3 -m src.main
