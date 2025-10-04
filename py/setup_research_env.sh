#!/bin/bash
# Development Environment Setup Script for HMM Research Environment
# This script sets up the complete research environment with all dependencies

set -e  # Exit on error

echo "=========================================="
echo "IMP Research Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_status "Python version $PYTHON_VERSION is compatible (>= $REQUIRED_VERSION)"
else
    print_error "Python version $PYTHON_VERSION is not compatible. Required: >= $REQUIRED_VERSION"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv .venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_status "pip upgraded"

# Install package with all dependencies
echo ""
echo "Installing IMP package with research dependencies..."
pip install -e ".[dev,optimization,research]"
print_status "Package installed with all dependencies"

# Setup Jupyter kernel
echo ""
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=imp-research --display-name="IMP Research Environment"
print_status "Jupyter kernel 'imp-research' installed"

# Install Jupyter extensions
echo ""
echo "Installing Jupyter extensions..."
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build 2>/dev/null || true
jupyter lab build --minimize=False 2>/dev/null || print_warning "JupyterLab build skipped (optional)"
print_status "Jupyter extensions configured"

# Validate installation
echo ""
echo "Validating installation..."
python -c "
import sys
import importlib.util

required_packages = [
    'numpy', 'pandas', 'polars', 'sklearn', 'hmmlearn', 'pomegranate',
    'jupyter', 'ipywidgets', 'matplotlib', 'seaborn', 'plotly',
    'pydantic', 'pytest'
]

missing = []
for package in required_packages:
    # Handle special cases
    pkg_name = 'scikit-learn' if package == 'sklearn' else package
    spec = importlib.util.find_spec(package)
    if spec is None:
        missing.append(pkg_name)

if missing:
    print(f'Missing packages: {', '.join(missing)}')
    sys.exit(1)
else:
    print('All required packages are installed')
"

if [ $? -eq 0 ]; then
    print_status "All dependencies validated successfully"
else
    print_error "Some dependencies are missing"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p notebooks/utils
mkdir -p notebooks/processed_data
mkdir -p notebooks/model_comparison_results
mkdir -p notebooks/regime_analysis_results
mkdir -p processed_data
mkdir -p temp_configs
print_status "Directory structure created"

# Run environment validation
echo ""
echo "Running comprehensive environment validation..."
python -m imp.utils.env_validator
if [ $? -eq 0 ]; then
    print_status "Environment validation passed"
else
    print_warning "Environment validation completed with warnings"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Start Jupyter Lab: jupyter lab"
echo "  3. Select kernel: 'IMP Research Environment'"
echo "  4. Open notebooks in notebooks/ directory"
echo ""
echo "For more information, see py/docs/DEVELOPMENT_SETUP.md"
echo ""
