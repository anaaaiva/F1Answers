FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .env
COPY src ./src
COPY data ./data

CMD ["streamlit run ./src/app.py"]
