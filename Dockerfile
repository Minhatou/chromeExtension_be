# Use a lightweight official Python runtime as a parent image
FROM python:3.11-slim


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for EasyOCR, PyTorch, and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# We also install gunicorn for production wsgi serving
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Pre-download EasyOCR models (vi, en) during build so they are cached in the image
RUN python -c "import easyocr; easyocr.Reader(['vi', 'en'])"

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Start the Flask app using Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app"]
