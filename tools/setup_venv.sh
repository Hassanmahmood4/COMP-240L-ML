#!/bin/bash
# Lab Task 5 helper: create a virtual environment and install dependencies.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$LAB_ROOT"

echo "Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing packages: scikit-learn, matplotlib, pandas (+ other lab deps)..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Virtual environment ready."
echo "Activate it with:  source .venv/bin/activate"
echo "Then run scripts, e.g.:  python \"Lab 2/task5_visualization.py\""
