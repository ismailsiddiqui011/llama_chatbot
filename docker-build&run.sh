# Build the Docker image
docker build -t my-ollama-app .

# Run the container with required ports
docker run --rm -p 8000:8000 -p 11434:11434 my-ollama-app