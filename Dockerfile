# Use a slim Python runtime as a base image
FROM python:3.9-slim-buster AS build

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app
COPY . /app

# Command to run the Python application
CMD ["python", "multi-test.py"]
#CMD ["python", "test_genome.py"]