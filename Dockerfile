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
COPY entrypoint.sh start.sh ./
RUN chmod +x entrypoint.sh start.sh

# Run the startup script when the container launches
# For Railway, use start.sh; for Docker directly, use entrypoint.sh
CMD ["sh", "start.sh"]