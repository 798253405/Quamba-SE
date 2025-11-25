#!/bin/bash

# ============================================================================
# 捕获所有modes的第一层输入输出
# ============================================================================

PRETRAINED_DIR="pretrained_models/Quamba1-pa9999/pa-0.9999"
MAMBA_MODEL_PATH="pretrained_models/mambaOriginalHuggingfaceDownload"
OUTPUT_FILE="first_layer_io_all_modes.npz"
SEQ_LEN=512

echo "============================================================================"
echo "Capturing First Layer I/O for All Modes"
echo "============================================================================"
echo ""
echo "Quamba pretrained dir: ${PRETRAINED_DIR}"
echo "Mamba model path:      ${MAMBA_MODEL_PATH}"
echo "Output file:           ${OUTPUT_FILE}"
echo "Sequence length:       ${SEQ_LEN}"
echo ""

# Parse arguments
if [ "$1" = "essential" ]; then
    MODES="1 fp32 0 2-1 2-2 2-4"
    echo "Running essential modes only: ${MODES}"
elif [ "$1" = "quant_only" ]; then
    MODES="0 2-0 2-1 2-2 2-3 2-4 3"
    echo "Running quantized modes only: ${MODES}"
elif [ $# -gt 0 ]; then
    MODES="$@"
    echo "Running custom modes: ${MODES}"
else
    MODES="1 fp32 0 2-0 2-1 2-2 2-3 2-4 3"
    echo "Running all modes (including mode 1): ${MODES}"
fi

echo ""

# Run the Python script
python3 save_first_layer_io.py \
    --pretrained_dir ${PRETRAINED_DIR} \
    --mamba_model_path ${MAMBA_MODEL_PATH} \
    --modes ${MODES} \
    --output_file ${OUTPUT_FILE} \
    --seq_len ${SEQ_LEN}

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ First layer I/O captured successfully!"
    echo "============================================================================"
    echo ""
    echo "Output files:"
    echo "  ${OUTPUT_FILE}                    # Full data (numpy arrays)"
    echo "  ${OUTPUT_FILE%.npz}_stats.json    # Statistics (JSON)"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "✗ Failed to capture first layer I/O"
    echo "============================================================================"
    exit 1
fi
