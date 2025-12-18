#!/bin/bash

# 1. Start FastAPI (Backend)
# Use 0.0.0.0 to ensure it listens on all internal interfaces
echo "🚀 Starting FastAPI Backend..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# 2. Wait LONGER for the Brain to load heavy models
echo "⏳ Waiting 10 seconds for models to load..."
sleep 10

# 3. Start Streamlit
echo "🎨 Starting Streamlit Frontend on port $PORT..."
streamlit run src/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0