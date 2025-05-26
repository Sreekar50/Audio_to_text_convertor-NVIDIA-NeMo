FROM python:3.12-slim

WORKDIR /app

COPY deploy_requirements.txt .
RUN pip install --no-cache-dir -r deploy_requirements.txt

COPY app/ app/
COPY model/ model/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]