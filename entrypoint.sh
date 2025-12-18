#!/bin/bash

# 1. Start FastAPI (Backend) in the background
# We run it on localhost:8000 so the internal dashboard can find it.
echo "🚀 Starting FastAPI Backend..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

# 2. Wait a few seconds for the Brain to wake up
sleep 5

# 3. Start Streamlit (Frontend)
# Railway automatically assigns a port number to the $PORT variable.
# We MUST tell Streamlit to listen on that specific port.
echo "🎨 Starting Streamlit Frontend on port $PORT..."
streamlit run src/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0