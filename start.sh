#!/bin/bash

# 1. Start the API in the background (&) on port 8000
# This runs silently so the script can continue to the next line
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Start the Dashboard in the foreground on port 7860
# This keeps the container running and visible to the user
streamlit run dashboard/frontend.py --server.port 7860 --server.address 0.0.0.0