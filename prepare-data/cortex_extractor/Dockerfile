FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash pipelineuser
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY export-btc.py validator.py /app/

RUN mkdir -p /tmp/parquet_stage && chown -R pipelineuser:pipelineuser /tmp/parquet_stage
ENV TMPDIR=/tmp/parquet_stage

USER pipelineuser

ENTRYPOINT ["python3", "export-btc.py"]
