#!/bin/sh
# Startup script for Railway deployment
# Reads PORT from environment and starts uvicorn

# Get PORT from environment, default to 8080
PORT="${PORT:-8080}"

# Validate PORT is a number
if ! echo "$PORT" | grep -qE '^[0-9]+$'; then
    echo "ERROR: PORT must be a number, got: $PORT"
    exit 1
fi

echo "Starting server on port $PORT"

# Start uvicorn
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"

