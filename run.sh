#!/bin/bash
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
# Run 'ollama serve' in the background
ollama serve &

# Run 'ollama run llama3.2' in the background
ollama run llama3.2 &

# Run the Python API in the foreground
python api.py
