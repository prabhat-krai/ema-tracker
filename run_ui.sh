#!/usr/bin/env bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Activate virtual environment and start Streamlit app
source venv/bin/activate
streamlit run src/app.py --browser.gatherUsageStats=false --server.headless=true
