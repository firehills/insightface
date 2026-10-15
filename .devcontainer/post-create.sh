#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m pip install -e './python-package[gui]'
python -m pip check
python -c 'import cv2, insightface, onnxruntime; print("ONNX Runtime providers:", onnxruntime.get_available_providers())'