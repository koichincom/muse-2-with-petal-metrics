#!/bin/bash

# Set the virtual environment directory name
VENV_DIR=".venv"

# Check if virtual environment exists, if not create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    echo "Virtual environment created in $VENV_DIR"
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install numpy matplotlib pylsl scipy

echo "All dependencies installed."
echo "To activate the environment, run: source $VENV_DIR/bin/activate"