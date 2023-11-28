FROM python:3.10.9-slim

WORKDIR /usr/src/app

RUN pip install mlflow==2.8.0 --no-cache-dir

EXPOSE 8080

CMD mlflow server --host 0.0.0.0 --port 8080
