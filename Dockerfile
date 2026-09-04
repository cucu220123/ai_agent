FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data ./data
COPY docs ./docs
COPY scripts ./scripts
COPY tests ./tests
COPY README.md ./.env.example ./
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

