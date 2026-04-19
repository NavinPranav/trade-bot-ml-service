FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./app/grpc_server/generated \
    --grpc_python_out=./app/grpc_server/generated \
    ./protos/prediction_service.proto \
    && sed -i 's/^import prediction_service_pb2 as/from . import prediction_service_pb2 as/' \
        ./app/grpc_server/generated/prediction_service_pb2_grpc.py

EXPOSE 50051 8081

CMD ["python", "-m", "app.main"]