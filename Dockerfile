FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --pre mlflow

EXPOSE 8080

# ENTRYPOINT [ "tail", "-f", "/dev/null" ]
ENTRYPOINT [ "mlflow", "server", "--host", "0.0.0.0", "--port", "8080" ]