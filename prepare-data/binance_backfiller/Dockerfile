
FROM python:3.10-slim


WORKDIR /app




COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY detect_gaps.py .
COPY extract_gaps.py .
COPY recreate_24h_ticker_for_gap.py .
COPY upload_data.py .
COPY run_pipeline.py .


CMD ["python", "run_pipeline.py"]
