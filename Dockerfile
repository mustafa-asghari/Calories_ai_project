# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup scripts executable
RUN chmod +x start.sh start.py

# Expose port (Railway will set PORT env var)
EXPOSE 8080

# Set default PORT if not provided (Railway will override this)
ENV PORT=8080

# Run the application using Python startup script
# This ensures PORT is properly read from environment
CMD ["python", "start.py"]

