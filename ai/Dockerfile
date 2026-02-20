FROM python:3.13-slim

WORKDIR /app

# Install AI dependencies
COPY ai/requirements.txt /tmp/ai-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ai-requirements.txt

# Copy AI module
COPY ai/ /app/ai/

WORKDIR /app/ai
