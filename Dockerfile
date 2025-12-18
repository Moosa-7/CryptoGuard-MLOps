# 1. Base Image
FROM python:3.12-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies
# REMOVED: software-properties-common (caused build failure and is not strictly needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY . .

# 6. Expose Hugging Face's default port
EXPOSE 7860

# 7. Command to run BOTH FastAPI (Background) and Streamlit (Foreground)
CMD uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run src/ui/dashboard.py --server.port 7860 --server.address 0.0.0.0