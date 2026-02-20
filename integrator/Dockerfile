FROM python:3.13-slim

WORKDIR /app

COPY integrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy integrator code
COPY integrator/ /app/
# Copy AI module (needed by ai_bridge)
COPY ai/ /app/ai/
# Copy .env
COPY .env /app/.env

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8001"]
