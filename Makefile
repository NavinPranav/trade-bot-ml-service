PROTO_DIR = protos
GENERATED_DIR = app/grpc_server/generated
# Prefer project venv so grpc_tools.protoc is available (Homebrew python3 may lack it).
PROTO_PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: proto run train prepare-training-data test clean

proto:
	$(PROTO_PYTHON) -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(GENERATED_DIR) \
		--grpc_python_out=$(GENERATED_DIR) \
		$(PROTO_DIR)/prediction_service.proto
	@sed -i '' 's/^import prediction_service_pb2 as/from . import prediction_service_pb2 as/' $(GENERATED_DIR)/prediction_service_pb2_grpc.py 2>/dev/null || sed -i 's/^import prediction_service_pb2 as/from . import prediction_service_pb2 as/' $(GENERATED_DIR)/prediction_service_pb2_grpc.py
	@touch $(GENERATED_DIR)/__init__.py
	@echo "Proto stubs generated in $(GENERATED_DIR)"

run:
	python3 -m app.main

prepare-training-data:
	python scripts/prepare_training_data.py

train:
	python scripts/train_models.py

test:
	pytest tests/ -v

celery-worker:
	celery -A app.scheduler.celery_config worker --loglevel=info

celery-beat:
	celery -A app.scheduler.celery_config beat --loglevel=info

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache