#!/bin/bash
# Quick test script for ProtT5 config files

echo "================================================"
echo "ProtT5 Config Files Validation"
echo "================================================"
echo ""

# Check if test data exists
if [ ! -f "test_data_aa.fasta" ]; then
    echo "❌ Error: test_data_aa.fasta not found"
    exit 1
fi

if [ ! -f "test_data_3di_masked.fasta" ]; then
    echo "❌ Error: test_data_3di_masked.fasta not found"
    exit 1
fi

echo "✓ Test data files found"
echo ""

# Validate config files
echo "Validating config files..."
for config in config_prott5_minimal.json config_prott5_test.json config_prott5_advanced.json; do
    if python3 -c "import json; json.load(open('$config'))" 2>/dev/null; then
        params=$(python3 -c "import json; print(len(json.load(open('$config'))))")
        echo "  ✓ $config ($params parameters)"
    else
        echo "  ❌ $config - Invalid JSON"
        exit 1
    fi
done
echo ""

echo "================================================"
echo "Config files are ready to use!"
echo "================================================"
echo ""
echo "To start training with ProtT5:"
echo ""
echo "  # Minimal (low memory)"
echo "  python -m esm3di.esmretrain --config config_prott5_minimal.json"
echo ""
echo "  # Standard test"
echo "  python -m esm3di.esmretrain --config config_prott5_test.json"
echo ""
echo "  # Advanced features"
echo "  python -m esm3di.esmretrain --config config_prott5_advanced.json"
echo ""
echo "See PROTT5_CONFIGS.md for detailed documentation."
