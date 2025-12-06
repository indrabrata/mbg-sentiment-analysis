#!/bin/bash

set -e

echo "=========================================="
echo "MLflow Deployment Setup"
echo "=========================================="
echo ""

if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and change the default passwords!"
    echo "   Run: nano .env"
    echo ""
    read -p "Press Enter after updating .env file..."
else
    echo "✓ .env file already exists"
fi

echo ""
echo "Starting Docker services..."
docker compose up -d

echo ""
echo "Waiting for services to be healthy..."
sleep 15

echo ""
echo "Checking service status..."
docker compose ps

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="

