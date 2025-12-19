#!/bin/sh
# Start script with proper PORT handling for Railway/Heroku deployments
# Runs both FastAPI backend and Streamlit frontend

# Set default port if not provided
if [ -z "$PORT" ]; then
    PORT=8501
fi
export PORT

# Start FastAPI backend in background on port 8000
echo "🚀 Starting FastAPI Backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a bit for API to start
sleep 3

# Check if API started successfully
if ! kill -0 $API_PID 2>/dev/null; then
    echo "⚠️ Warning: FastAPI backend may not have started correctly"
else
    echo "✅ FastAPI Backend started (PID: $API_PID)"
fi

# Start Streamlit frontend on the provided PORT
echo "🎨 Starting Streamlit Frontend on port $PORT..."
exec streamlit run src/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0
