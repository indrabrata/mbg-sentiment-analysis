#!/bin/bash

set -e

BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "MLflow Deployment Backup"
echo "=========================================="
echo ""

mkdir -p "$BACKUP_DIR"

echo "Backing up PostgreSQL database..."
docker-compose exec -T postgres pg_dump -U mlflow mlflow > "$BACKUP_DIR/postgres_backup_$TIMESTAMP.sql"
echo "✓ PostgreSQL backup saved to: $BACKUP_DIR/postgres_backup_$TIMESTAMP.sql"

echo "Backing up .env file..."
cp .env "$BACKUP_DIR/env_backup_$TIMESTAMP"
echo "✓ Environment backup saved to: $BACKUP_DIR/env_backup_$TIMESTAMP"

cat > "$BACKUP_DIR/backup_info_$TIMESTAMP.txt" <<EOF
Backup created: $(date)
PostgreSQL backup: postgres_backup_$TIMESTAMP.sql
Environment backup: env_backup_$TIMESTAMP

To restore PostgreSQL:
  cat postgres_backup_$TIMESTAMP.sql | docker-compose exec -T postgres psql -U mlflow mlflow

MinIO data is stored in Docker volume: github-action-deployment_minio_data
To backup MinIO volume:
  docker run --rm -v github-action-deployment_minio_data:/data -v \$(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/minio_backup_$TIMESTAMP.tar.gz -C /data .

To restore MinIO volume:
  docker run --rm -v github-action-deployment_minio_data:/data -v \$(pwd)/$BACKUP_DIR:/backup alpine tar xzf /backup/minio_backup_$TIMESTAMP.tar.gz -C /data
EOF

echo "✓ Backup info saved to: $BACKUP_DIR/backup_info_$TIMESTAMP.txt"

echo ""
echo "=========================================="
echo "Backup Complete!"
echo "=========================================="
echo ""
echo "Backup files created:"
echo "  - PostgreSQL: $BACKUP_DIR/postgres_backup_$TIMESTAMP.sql"
echo "  - Environment: $BACKUP_DIR/env_backup_$TIMESTAMP"
echo "  - Info: $BACKUP_DIR/backup_info_$TIMESTAMP.txt"
echo ""
echo "Note: MinIO data in Docker volume was not backed up."
echo "See backup info file for MinIO backup commands."
echo ""
