FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# --- THE FIX: Copy the pre-downloaded model files ---
COPY ./model_files ./model_files

# Copy the rest of the app
COPY . .

EXPOSE 5002
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--timeout", "600", "app:app"]