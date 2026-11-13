#!/bin/bash
#
# SecureFed-NIDS Setup and Run Script
# ====================================
# This script:
# 1. Sets up the virtual environment with required dependencies
# 2. Runs the full training pipeline
# 3. Starts the HQ server and dashboard
# 4. Provides client interface
#
# Usage: ./setup_and_run.sh [mode]
#
# Modes:
#   setup     - Install dependencies and prepare environment (Default)
#   pipeline  - Run the full 7-stage training pipeline
#   hq        - Start HQ server and dashboard only
#   client    - Start client interface only
#   all       - Setup + Pipeline + HQ (default)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root (auto-detected directory of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
VENV_PATH="$PROJECT_ROOT/venv"

echo "======================================================"
echo "        SecureFed-NIDS Setup and Run Script"
echo "======================================================"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function: Check Python version
check_python() {
    print_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version)
        print_success "Python found: $PYTHON_VERSION"
        return 0
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        PYTHON_VERSION=$(python --version)
        print_success "Python found: $PYTHON_VERSION"
        return 0
    else
        print_error "Python 3.7 or later is required"
        return 1
    fi
}

# Function: Setup virtual environment
setup_venv() {
    print_info "Setting up virtual environment..."

    if [ -d "$VENV_PATH" ]; then
        print_warning "Virtual environment already exists. Recreating..."
        rm -rf "$VENV_PATH"
    fi

    $PYTHON_CMD -m venv "$VENV_PATH"

    # Fix venv python symlink
    mkdir -p "$VENV_PATH/bin"
    ln -sfn $(which $PYTHON_CMD) "$VENV_PATH/bin/python"
    ln -sfn python "$VENV_PATH/bin/python3"

    print_success "Virtual environment created at: $VENV_PATH"
}

# Function: Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."

    "$VENV_PATH/bin/pip" install --upgrade pip --break-system-packages -q

    "$VENV_PATH/bin/pip" install \
        fastapi \
        uvicorn \
        jinja2 \
        requests \
        torch \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        matplotlib \
        seaborn \
        mlflow \
        python-dotenv \
        shap \
        psutil \
        python-multipart \
        --break-system-packages -q

    print_success "Dependencies installed"
}

# Function: Run training pipeline
run_pipeline() {
    print_info "Running 7-stage training pipeline..."

    "$VENV_PATH/bin/python" main.py --mode all --epochs 3

    print_success "Pipeline completed successfully"
}

# Function: Start HQ server and dashboard
start_hq() {
    print_info "Starting HQ server and dashboard..."
    "$VENV_PATH/bin/python" start_hq.py
}

# Function: Start client interface
start_client() {
    print_info "Starting Edge Client Node..."
    exec ./run_client.sh 5000
}

# Function: Run everything
run_all() {
    check_python || exit 1
    setup_venv
    install_dependencies
    run_pipeline
    start_hq
}

# Main
MODE="${1:-all}"

case "$MODE" in
    setup)
        check_python || exit 1
        setup_venv
        install_dependencies
        ;;
    pipeline)
        check_python || exit 1
        run_pipeline
        ;;
    hq)
        start_hq
        ;;
    client)
        start_client
        ;;
    all)
        run_all
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo "Usage: $0 [setup|pipeline|hq|client|all]"
        exit 1
        ;;
esac
