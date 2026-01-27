#!/bin/bash
# Setup Virtual Environment untuk SimpleNeural-DSL
# Untuk Arch Linux dengan externally-managed-environment

echo "════════════════════════════════════════════════════════════════"
echo "  🐍 SimpleNeural-DSL - Virtual Environment Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Venv in project folder
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "📁 Project directory: $PROJECT_DIR"
echo "📦 Venv location: $VENV_DIR"
echo ""

# Detect Python 3.13
echo "🔍 Looking for Python 3.13..."
PYTHON_CMD=""

# Try to find Python 3.13
for py_version in python3.13 python3; do
    if command -v $py_version &>/dev/null; then
        PY_VER=$($py_version --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
        echo "   Found: $py_version (version $PY_VER)"
        
        if [[ "$PY_VER" == "3.13" ]] || [[ "$PY_VER" == "3.11" ]]; then
            PYTHON_CMD=$py_version
            echo "   ✅ Selected: $py_version"
            break
        fi
    fi
done

# If not found, show error
if [ -z "$PYTHON_CMD" ]; then
    echo "   ❌ Python 3.13 or 3.11 not found!"
    echo ""
    echo "💡 Install Python:"
    echo "   sudo pacman -S python  # If 3.13 is default"
    echo "   # or"
    echo "   yay -S python313       # From AUR"
    echo ""
    exit 1
fi

echo ""

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at: $VENV_DIR"
    read -p "Delete and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old venv..."
        rm -rf "$VENV_DIR"
    else
        echo "❌ Setup cancelled."
        exit 0
    fi
fi

echo "📦 Creating virtual environment with Python 3.13..."
$PYTHON_CMD -m venv "$VENV_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment!"
    exit 1
fi

echo "✅ Virtual environment created!"
echo "   Python: $($PYTHON_CMD --version)"
echo ""

# Activate venv
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "✅ Activated!"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo "📦 Installing dependencies..."
echo ""

echo "  • Installing kaggle..."
pip install kaggle -q

echo "  • Installing TensorFlow..."
# For Python 3.13, try pip first
pip install tensorflow -q 2>/dev/null
if [ $? -eq 0 ]; then
    echo "    ✅ TensorFlow installed via pip"
else
    echo "    ⚠️  TensorFlow not available via pip for Python 3.13"
    echo "    💡 Options:"
    echo "       1. Wait for TensorFlow 3.13 support"
    echo "       2. Use nightly build: pip install tf-nightly"
    echo "       3. Use system package: sudo pacman -S python-tensorflow"
    MISSING_TF=true
fi

echo "  • Installing pandas..."
pip install pandas -q

echo "  • Installing numpy..."
pip install numpy -q

echo "  • Installing scikit-learn..."
pip install scikit-learn -q

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if TensorFlow is available
if [ "$MISSING_TF" = true ]; then
    echo "⚠️  TensorFlow Installation Required:"
    echo ""
    echo "   Option 1: Install nightly build (Recommended)"
    echo "   source venv/bin/activate"
    echo "   pip install tf-nightly"
    echo ""
    echo "   Option 2: Install system package"
    echo "   sudo pacman -S python-tensorflow"
    echo ""
fi

echo "📋 NEXT STEPS:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run UI:"
echo "   python ui.py"
echo ""
echo "3. When done, deactivate:"
echo "   deactivate"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "💡 TIP: Add this to your shell config for easy activation:"
echo ""
echo "   alias automata='cd $PROJECT_DIR && source venv/bin/activate'"
echo ""
echo "Then you can just type: automata"
echo ""
