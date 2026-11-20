FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY api ./api
COPY models ./models
COPY exports ./exports
COPY data/processed ./data/processed
COPY reports/model_results.json ./reports/model_results.json
COPY reports/figures ./reports/figures
EXPOSE 8501 8000
# default: Streamlit app.  API: docker run -p 8000:8000 agrisense uvicorn api.main:app --host 0.0.0.0
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
