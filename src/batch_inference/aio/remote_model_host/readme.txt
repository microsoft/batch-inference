# Run this command in source root to generate protobuf py files:
python -m grpc_tools.protoc -I src --python_out=src --pyi_out=src --grpc_python_out=src src/batch_inference/aio/remote_model_host/*.proto
