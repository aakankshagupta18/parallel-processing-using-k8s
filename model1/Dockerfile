# Use the official Python image as base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Define environment variable for the result file path (optional override)
ENV RESULT_FILE_PATH="/data/results.json"

# Run the Python script
CMD ["python3", "server.py"]
