FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV and other heavy libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose port (Render/Railway will provide their own, but default to 5000 locally)
EXPOSE 5000

# Start Gunicorn server (since Flask is not meant for production directly)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
