#!/bin/bash

# Jetson-specific setup script (Safe version for existing conda environment)
# Only installs missing dependencies, preserves existing PyTorch installation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Safe Setup for Existing foundationpose_ga Environment ==="
echo ""
echo "✓ This script will:"
echo "  - Keep your existing PyTorch 2.3.0 (CUDA 12.4)"
echo "  - Keep your existing TensorRT 10.3.0"
echo "  - Only install missing dependencies"
echo ""

# Check if we're in the right conda environment
if [[ "$CONDA_DEFAULT_ENV" != "foundationpose_ga" ]]; then
    echo "⚠️  Warning: You should activate foundationpose_ga environment first:"
    echo "   conda activate foundationpose_ga"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check existing packages
echo "Checking existing packages..."
EXISTING_PACKAGES=$(pip list)

# Function to check if package is installed
is_installed() {
    echo "$EXISTING_PACKAGES" | grep -q "^$1 "
}

# Install missing basic dependencies
echo ""
echo "Installing missing basic dependencies..."
MISSING_BASIC=""
is_installed "imageio" || MISSING_BASIC="$MISSING_BASIC imageio"
is_installed "open3d" || MISSING_BASIC="$MISSING_BASIC open3d"

if [ -n "$MISSING_BASIC" ]; then
    pip install $MISSING_BASIC
else
    echo "✓ All basic dependencies already installed"
fi

# Install warp-lang if missing
if ! is_installed "warp-lang"; then
    echo "Installing warp-lang..."
    pip install warp-lang==1.0.2
else
    echo "✓ warp-lang already installed"
fi

# Install kornia if missing
if ! is_installed "kornia"; then
    echo "Installing kornia..."
    pip install kornia==0.7.2
else
    echo "✓ kornia already installed"
fi

# Install nvdiffrast if missing
if ! is_installed "nvdiffrast"; then
    echo "Installing nvdiffrast..."
    pip install git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8
else
    echo "✓ nvdiffrast already installed"
fi

# Install pytorch3d if missing (requires CUDA environment)
if ! is_installed "pytorch3d"; then
    echo "Installing pytorch3d (this may take a while)..."
    source "${SCRIPT_DIR}/deps_jetson.sh"
    activate_deps
    pip install git+https://github.com/facebookresearch/pytorch3d.git@d098beb7a7f92ee226de97b1b7905ee735aeed56 --no-build-isolation
    deactivate_deps
else
    echo "✓ pytorch3d already installed"
fi

# Install the package itself
echo ""
echo "Installing FoundationPose-TensorRT package..."
pip install -e "$SCRIPT_DIR/.."

# Check numpy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "Downgrading numpy to <2.0 for compatibility..."
    pip uninstall numpy -y
    pip install "numpy<2"
else
    echo "✓ numpy version is compatible: $NUMPY_VERSION"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Your environment summary:"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "  - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  - TensorRT: $(python -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || echo 'Not found')"
echo ""
echo "To use the project, activate dependencies:"
echo "  source scripts/deps_jetson.sh && activate_deps"
