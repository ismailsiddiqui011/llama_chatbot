#!/bin/bash

# Log and install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Dependencies installed."

# Log and install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
echo "Ollama installation completed."

# Log and run 'ollama serve' in the background
echo "Starting 'ollama serve' in the background..."
ollama serve &
echo "'ollama serve' started."

# Log and run 'ollama run llama3.2' in the background
echo "Starting 'ollama run llama3.2' in the background..."
ollama run llama3.2 &
echo "'ollama run llama3.2' started."

# Log and run the Python API in the foreground
echo "Starting Python API (api.py) in the foreground..."
python api.py
