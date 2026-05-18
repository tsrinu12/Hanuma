# Cold-start bandit service.
# Built from the repo root so `shared/` is in scope.
#   docker build -f services/cold_start_bandit/Dockerfile -t distrebute/cold-start-bandit .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Layer caching: install deps first.
COPY services/cold_start_bandit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code.
COPY shared /app/shared
COPY services/cold_start_bandit/app /app/app

ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
