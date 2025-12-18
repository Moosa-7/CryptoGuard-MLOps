# Use lightweight Python 3.12
FROM python:3.12-slim

# Install system tools (needed for building some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Copy and set permissions for the startup script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run the startup script when the container launches
CMD ["./entrypoint.sh"]