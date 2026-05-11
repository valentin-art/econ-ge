#!/bin/bash
# File: wait_for_docker.sh
# This script waits until Docker Desktop is ready in WSL

# Optional: timeout in seconds
TIMEOUT=30
INTERVAL=2
ELAPSED=0

echo "Waiting for Docker to be ready..."

while true; do
    if docker info >/dev/null 2>&1; then
        echo "Docker is ready!"
        break
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Timeout: Docker did not become ready within $TIMEOUT seconds."
        exit 1
    fi
done
