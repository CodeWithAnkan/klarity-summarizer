# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code (app.py) into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5002

# The command to run your application
CMD ["python", "app.py"]