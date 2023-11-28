docker build -t ai-backend-image -f docker/Dockerfile .
docker build -t ai-mlflow-server-image -f docker/mlflow/Dockerfile .
docker compose -f docker/docker-compose.yml up -d
