#!/usr/bin/env bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run quantitative momentum portfolio screener CLI passing all user arguments
python3 -m src.momentum "$@"
