#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Kill any existing services on these ports
echo "Cleaning up existing services..."
pkill -f "app.py" 2>/dev/null
pkill -f "server.py" 2>/dev/null
pkill -f "python.*8080" 2>/dev/null
pkill -f "python.*3000" 2>/dev/null
sleep 2

clear
echo "Starting Agentic RAG Services..."

# Start backend service in background (suppress output)
export FLASK_APP=backend/app.py
export FLASK_ENV=development
python3 backend/app.py > /dev/null 2>&1 &
BACKEND_PID=$!

# Start frontend service in background (suppress output)
cd frontend
export FLASK_ENV=development
python3 server.py > /dev/null 2>&1 &
FRONTEND_PID=$!

# Go back to root directory
cd ..

# Wait a moment for services to start
sleep 3

# Check if services are running
if kill -0 $BACKEND_PID 2>/dev/null && kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "Services successfully started!"
    echo ""
    echo "Frontend: http://localhost:3000"
    echo "Backend:  http://localhost:8080"
    echo ""
    echo "Open the frontend link in your browser"
    echo "Press Ctrl+C to stop both services"
else
    echo "ERROR: Failed to start services"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    # Also kill any remaining processes
    pkill -f "app.py" 2>/dev/null
    pkill -f "server.py" 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT

# Wait for background processes
wait