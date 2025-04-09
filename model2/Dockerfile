# Use a base Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy script and install dependencies
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the script
CMD ["python3", "model.py"]
