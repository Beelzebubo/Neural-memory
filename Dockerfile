# ============================================================
# Lino — Dockerfile
# Multi-stage: build venv in stage 1, runtime in stage 2 (slim)
# ============================================================

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

RUN useradd -m -u 1000 lino && mkdir -p /data /logs /config && chown -R lino:lino /data /logs /config

COPY --from=builder /root/.local /home/lino/.local
COPY --chown=lino:lino src/ /app/src/
COPY --chown=lino:lino ui/ /app/ui/
COPY --chown=lino:lino config/ /app/config/
COPY --chown=lino:lino integration/ /app/integration/

ENV PATH=/home/lino/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STORE_PATH=/data/memory_store.pkl \
    LOG_FILE=/logs/lino.log

WORKDIR /app
USER lino

EXPOSE 8210

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8210/ui/')" || exit 1

CMD ["python3", "-m", "uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8210"]
