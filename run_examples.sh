#!/bin/bash
# SimpleNeural-DSL - Command Examples
# Demonstrasi penggunaan CLI

echo "=========================================="
echo "SimpleNeural-DSL CLI - Usage Examples"
echo "=========================================="
echo ""

# Check version
echo "1. Check version:"
echo "   $ simpleneural --version"
python -m simpleneural --version
echo ""

# Validate examples
echo "2. Validate DSL files:"
echo "   $ simpleneural validate examples/minimal.sndsl"
python -m simpleneural validate examples/minimal.sndsl
echo ""

echo "   $ simpleneural validate examples/housing_regression.sndsl"
python -m simpleneural validate examples/housing_regression.sndsl
echo ""

# Compile example
echo "3. Compile DSL to Python:"
echo "   $ simpleneural compile examples/minimal.sndsl -o output.py"
python -m simpleneural compile examples/minimal.sndsl -o compiled_minimal.py
echo ""

# Tokenize for debugging
echo "4. Debug: View tokens"
echo "   $ simpleneural tokenize examples/minimal.sndsl"
python -m simpleneural tokenize examples/minimal.sndsl | head -20
echo "   ... (output truncated)"
echo ""

# Show generated code snippet
echo "5. Generated Python code preview:"
echo "   First 30 lines of compiled_minimal.py:"
echo "   ----------------------------------------"
head -30 compiled_minimal.py
echo "   ... (truncated)"
echo ""

# Run tests
echo "6. Run test suite:"
echo "   $ python test_compiler.py"
python test_compiler.py | grep -E "(TEST|PASS|FAIL|Results)"
echo ""

echo "=========================================="
echo "All commands executed successfully!"
echo "=========================================="
echo ""
echo "Try these commands yourself:"
echo "  - simpleneural compile <file.sndsl>"
echo "  - simpleneural validate <file.sndsl>"
echo "  - simpleneural run <file.sndsl>"
echo "  - simpleneural --help"
echo ""
