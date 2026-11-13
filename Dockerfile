
# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from generating bytecode
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs appear immediately
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Copy exported model and metadata
COPY deployment/ ./deployment/

# Expose API port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
