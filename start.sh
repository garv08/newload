#!/bin/bash
# Startup script for the PHY Layer Multi-Agent Architecture application

echo "Installing dependencies for the PHY Layer Multi-Agent Architecture application..."
npm install --no-audit --no-fund --progress=false || echo "Installation failed, trying to run with existing packages..."

echo "Starting the PHY Layer Multi-Agent Architecture application..."
echo "Access the application at http://localhost:3000 or http://21.0.10.145:3000"

# Start the Vite development server
npx --yes vite