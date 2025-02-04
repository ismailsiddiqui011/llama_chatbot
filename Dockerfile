# Use an official Python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY api.py .
COPY run.sh /run.sh

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Make the script executable
RUN chmod +x /run.sh

# Expose necessary ports (adjust as needed)
EXPOSE 8000 11434

# Run the startup script
CMD ["/bin/bash", "/run.sh"]
