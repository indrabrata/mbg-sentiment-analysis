#!/bin/bash

# Airflow Permission Fixer Script
# This script fixes the ownership issues caused by Airflow Docker containers

set -e

echo "🔧 Airflow Permission Fixer"
echo "=========================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CURRENT_USER=$(whoami)
CURRENT_UID=$(id -u)

echo "👤 Current user: $CURRENT_USER (UID: $CURRENT_UID)"
echo ""

FOLDERS=("dags" "logs" "plugins" "config")

echo "📁 Checking folders..."
NEEDS_FIX=false
for folder in "${FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        OWNER=$(stat -c '%U' "$folder" 2>/dev/null || stat -f '%Su' "$folder" 2>/dev/null)
        if [ "$OWNER" != "$CURRENT_USER" ]; then
            echo -e "${RED}✗${NC} $folder (owned by: $OWNER)"
            NEEDS_FIX=true
        else
            echo -e "${GREEN}✓${NC} $folder"
        fi
    fi
done

echo ""

if [ "$NEEDS_FIX" = false ]; then
    echo -e "${GREEN}✅ All folders have correct permissions!${NC}"
    exit 0
fi

echo -e "${YELLOW}⚠️  Some folders need permission fixes.${NC}"
read -p "Do you want to fix permissions now? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled."
    exit 1
fi

echo ""
echo "🔐 Fixing permissions..."
echo ""

for folder in "${FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        echo "  → Fixing $folder..."
        sudo chown -R $CURRENT_USER:$CURRENT_USER "$folder"
    fi
done

echo ""
echo "📝 Setting proper permissions..."
chmod -R 755 dags/ plugins/ config/ data/ models/ scripts/ reports/ 2>/dev/null || true
chmod -R 777 logs/ 2>/dev/null || true

echo ""
echo -e "${GREEN}✅ All permissions fixed!${NC}"
echo ""

echo "📊 Final status:"
for folder in "${FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        OWNER=$(stat -c '%U' "$folder" 2>/dev/null || stat -f '%Su' "$folder" 2>/dev/null)
        echo -e "  ${GREEN}✓${NC} $folder → $OWNER"
    fi
done

echo ""
echo "🎉 Done! You can now work with your files."