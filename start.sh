#!/bin/bash
# Start script with proper PORT handling for Railway/Heroku deployments

# Set default port if not provided
export PORT=${PORT:-8501}

# Start Streamlit
streamlit run src/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0
