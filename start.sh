#!/bin/sh
# Start script with proper PORT handling for Railway/Heroku deployments

# Set default port if not provided
if [ -z "$PORT" ]; then
    PORT=8501
fi
export PORT

# Start Streamlit
exec streamlit run src/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0
