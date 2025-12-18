#!/bin/bash

# 1. FORCE TRAINING ON STARTUP (The "Nuclear Fix")
# This creates the real .pkl files inside the container, ignoring Git LFS issues.

echo "🧠 Training Bitcoin Model..."
python src/models/train_btc.py   # <--- Make sure this filename matches yours!

echo "👥 Training Segmentation Model..."
python src/models/train_segmentation.py   # <--- Make sure this filename matches yours!

# 2. Start FastAPI (Backend)
echo "🚀 Starting FastAPI Backend..."
# Use 0.0.0.0 to ensure it listens on all internal interfaces
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# 3. Wait for the Brain to wake up
echo "⏳ Waiting 5 seconds for backend..."
sleep 5

# 4. Start Streamlit (Frontend)
echo "🎨 Starting Streamlit Frontend on port $PORT..."
streamlit run src/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0