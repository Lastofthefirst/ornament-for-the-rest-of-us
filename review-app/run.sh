#!/bin/bash

# OCR Review App Launcher
# Starts both the Python backend and Vite dev server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default OCR output directory - can override with argument
OCR_DIR="${1:-../dots.ocr/output/rotated_final}"

# LLM settings - using local llamacpp server by default
export OCR_OUTPUT_DIR="$OCR_DIR"
export LLM_API_URL="${LLM_API_URL:-http://localhost:8080/v1/chat/completions}"
export LLM_MODEL="${LLM_MODEL:-gpt-oss-20b}"

echo "=== OCR Review App ==="
echo "OCR Directory: $OCR_OUTPUT_DIR"
echo "LLM API: $LLM_API_URL"
echo "LLM Model: $LLM_MODEL"
echo ""

# Check if OCR dir exists
if [ ! -d "$OCR_DIR" ]; then
    echo "Error: OCR output directory not found: $OCR_DIR"
    exit 1
fi

# Start backend server in background
echo "Starting backend server on port 8787..."
python server/server.py &
BACKEND_PID=$!

# Give backend time to start
sleep 2

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Backend failed to start"
    exit 1
fi

# Start frontend dev server
echo "Starting frontend on port 5173..."
echo ""
echo "Open http://localhost:5173 in your browser"
echo ""

pnpm dev

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null
