
FROM python:3.11-slim

# Prevent python from writing pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional; uncomment if you face build issues)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py ./
COPY features.json ./
COPY models/ ./models/

# Set models directory
ENV MODEL_DIR=/app/models
ENV FEATURES_PATH=/app/features.json
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
