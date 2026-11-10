#!/bin/bash
# Recreate virtual environment for SecureFed-NIDS

set -e

cd "$(dirname "$0")"

echo "Removing old venv..."
rm -rf venv

echo "Creating new venv with Python 3.10..."
python3 -m venv venv

echo "Upgrading pip..."
venv/bin/pip install --upgrade pip -q

echo "Installing dependencies..."
venv/bin/pip install \
    fastapi \
    uvicorn \
    jinja2 \
    numpy \
    pandas \
    scikit-learn \
    psutil \
    python-multipart \
    requests \
    torch \
    scipy \
    matplotlib \
    seaborn \
    mlflow \
    python-dotenv \
    shap \
    -q

echo "✓ Virtual environment recreated successfully!"
echo "✓ You can now run: venv/bin/python start_hq.py"
