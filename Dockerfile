# Use a slim Python runtime as a base image
FROM python:3.9-slim-buster AS build

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create directories and copy their content
RUN mkdir /app/trades
COPY trades /app/trades

RUN mkdir /app/src
COPY src /app/src

RUN mkdir /app/data
COPY data /app/data

RUN mkdir /app/config
COPY config /app/config

RUN mkdir /app/checkpoints
COPY config /app/checkpoints

# Copy the entire project directory into the container at /app
COPY . /app

# default Command to run the Python application without specifying parameter test
#CMD ["python", "multi-test.py"]
#CMD ["python", "test_genome.py"]
#CMD ["python", "new_main.py"]
CMD ["python", "live_trading_loop.py"]