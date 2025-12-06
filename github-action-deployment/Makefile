.PHONY: help setup start stop restart logs status backup clean down health

help:
	@echo "MLflow Deployment Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup    - Initial setup (copy .env, create directories, start services)"
	@echo "  make start    - Start all services"
	@echo "  make stop     - Stop all services"
	@echo "  make restart  - Restart all services"
	@echo "  make logs     - View logs from all services"
	@echo "  make status   - Check status of all services"
	@echo "  make backup   - Backup PostgreSQL database and configuration"
	@echo "  make health   - Check health of all services"
	@echo "  make clean    - Stop services and remove containers (keeps data)"
	@echo "  make down     - Stop and remove everything including volumes"
	@echo ""

setup:
	@./scripts/setup.sh

start:
	@echo "Starting all services..."
	@docker compose up -d
	@echo "Services started. Use 'make status' to check their status."

stop:
	@echo "Stopping all services..."
	@docker compose stop
	@echo "Services stopped."

restart:
	@echo "Restarting all services..."
	@docker compose restart
	@echo "Services restarted."

logs:
	@docker compose logs -f

status:
	@docker compose ps

backup:
	@./scripts/backup.sh

health:
	@echo "Checking service health..."
	@echo ""
	@echo "MLflow:"
	@curl -f http://localhost:5000/health 2>/dev/null && echo "  ✓ MLflow is healthy" || echo "  ✗ MLflow is not responding"
	@echo ""
	@echo "PostgreSQL:"
	@docker compose exec -T postgres pg_isready -U mlflow 2>/dev/null && echo "  ✓ PostgreSQL is healthy" || echo "  ✗ PostgreSQL is not responding"
	@echo ""
	@echo "MinIO:"
	@curl -f http://localhost:9000/minio/health/live 2>/dev/null && echo "  ✓ MinIO is healthy" || echo "  ✗ MinIO is not responding"
	@echo ""
	@echo "nginx:"
	@curl -f http://localhost:80/health 2>/dev/null && echo "  ✓ nginx is healthy" || echo "  ✗ nginx is not responding"
	@echo ""

clean:
	@echo "This will stop and remove containers but keep data volumes."
	@bash -c 'read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down; \
		echo "Containers removed. Data volumes preserved."; \
	else \
		echo "Cancelled."; \
	fi'

down:
	@echo "WARNING: This will delete ALL data including databases and artifacts!"
	@bash -c 'read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		echo "Everything removed including data volumes."; \
	else \
		echo "Cancelled."; \
	fi'
