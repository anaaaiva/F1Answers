FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target=/app/dependencies

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/dependencies /usr/local/lib/python3.11/site-packages/

COPY .env .
COPY src ./src
COPY data ./data

RUN python src/data_loading.py

CMD ["streamlit", "run", "./src/app.py"]
