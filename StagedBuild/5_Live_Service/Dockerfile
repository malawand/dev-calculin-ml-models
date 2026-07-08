FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ ./src/
COPY lib/ ./lib/
COPY artifacts/ ./artifacts/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app/lib

EXPOSE 8080 9109

CMD ["python", "src/service.py", "--config", "config.yaml"]
