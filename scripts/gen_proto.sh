#!/usr/bin/env bash
# Generate Python bindings from the proto contract.
# system protoc 不要 — grpcio-tools 同梱の protoc を使う(dev extra)。
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m grpc_tools.protoc \
  -I recolens/proto \
  --python_out=recolens/proto \
  recolens/proto/recolens.proto
echo "generated: recolens/proto/recolens_pb2.py"
