# Use an official Python slim runtime as a base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies (build-essential needed for compiling any wheels if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the application code
COPY app/ ./app
COPY data/ ./data

# Create data directories for vector store persistence
RUN mkdir -p data/vector_store data/resumes

# Expose the API port
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
