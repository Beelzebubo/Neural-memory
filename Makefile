.PHONY: start stop restart health backup restore seed install uninstall

HOST ?= 127.0.0.1
PORT ?= 8210

start:  ## Start Lino server (dev mode)
	./scripts/start.sh dev

stop:   ## Stop Lino server
	./scripts/stop.sh

restart: stop start  ## Restart Lino

prod:   ## Start in production mode
	./scripts/start.sh prod

health: ## Check server health
	./scripts/healthcheck.sh

backup: ## Backup memory store
	./scripts/backup.sh

restore: ## Restore from backup
	./scripts/restore.sh $(FILE)

seed:   ## Import Obsidian vault into Lino
	./scripts/seed.sh

install: ## Install systemd service
	sudo cp systemd/lino.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable lino.service
	sudo systemctl start lino.service
	@echo "Lino service installed and started."
	@echo "Status: sudo systemctl status lino.service"

uninstall: ## Remove systemd service
	sudo systemctl stop lino.service 2>/dev/null || true
	sudo systemctl disable lino.service 2>/dev/null || true
	sudo rm -f /etc/systemd/system/lino.service
	sudo systemctl daemon-reload
	@echo "Lino service uninstalled."

logs:   ## Tail server logs
	tail -f logs/lino.log 2>/dev/null || echo "No log file found"

docker: ## Build and run with Docker
	docker compose up --build -d

docker-stop: ## Stop Docker containers
	docker compose down

docker-logs: ## Tail Docker logs
	docker compose logs -f

venv:   ## Create virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

help:   ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
