# Use lightweight Python 3.12
FROM python:3.12-slim

# Install system tools (needed for building some ML libraries)
# REMOVED: software-properties-common (caused the build error)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Copy and set permissions for the startup scripts
COPY start.sh entrypoint.sh ./
RUN chmod +x start.sh entrypoint.sh

# Expose Streamlit port (Railway will set PORT env var)
EXPOSE 8501

# Run the startup script - it handles PORT env var properly
CMD ["./start.sh"]