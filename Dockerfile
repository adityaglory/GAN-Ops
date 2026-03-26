# Use an official Python runtime as a parent image (matching your Conda env)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (this optimizes Docker's caching system)
COPY requirements.txt .

# Install the Python dependencies
# We use --no-cache-dir to keep the Docker image size as small as possible
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the saved model
COPY . .

# Expose port 8000 so the outside world can communicate with the API
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]