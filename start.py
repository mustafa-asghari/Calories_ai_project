#!/usr/bin/env python3
"""Startup script for Railway deployment"""
import os
import uvicorn

# Get PORT from environment, default to 8080
port = int(os.environ.get("PORT", 8080))

print(f"Starting server on port {port}")

# Start uvicorn
uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=port,
    log_level="info"
)

