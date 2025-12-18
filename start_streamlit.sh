#!/bin/bash
# Start Streamlit with proper port handling
PORT=${PORT:-8501}
streamlit run src/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0

