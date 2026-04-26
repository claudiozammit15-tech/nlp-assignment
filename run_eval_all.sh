#!/bin/bash
# =============================================================================
# Full evaluation pipeline for D&R project
# Run this from the D-R-master directory inside WSL
#
# Prerequisites:
#   - Base GGUF:     ./models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf
#   - Distilled GGUF: ./models/qwen-distilled.gguf  (from convert_hf_to_gguf.py)
#   - History test data: ./data/test_data/MMLUProHist_test.json  (from gen_hist_data.py)
#   - conda activate dr_env
#
# This script runs 4 evaluations + 4 error analyses.
# It starts/stops the model server between base and distilled runs.
# =============================================================================

set -e  # Exit on error

PORT=58000
BASE_GGUF="./models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DISTILLED_GGUF="./models/qwen-distilled.gguf"

# Common test.py args
TEST_ARGS="--num_proc 1 --max_new_tokens 700 --temperature 0.3 --top_p 0.9 --port $PORT"

echo "============================================"
echo "PHASE 0: Pre-flight checks"
echo "============================================"

if [ ! -f "$BASE_GGUF" ]; then
    echo "ERROR: Base GGUF not found at $BASE_GGUF"
    exit 1
fi

if [ ! -f "$DISTILLED_GGUF" ]; then
    echo "ERROR: Distilled GGUF not found at $DISTILLED_GGUF"
    echo "Run the GGUF conversion first:"
    echo "  python llama.cpp/convert_hf_to_gguf.py ./models/qwen-distilled-merged --outfile ./models/qwen-distilled.gguf --outtype f16"
    exit 1
fi

if [ ! -f "./data/test_data/MMLUProHist_test.json" ]; then
    echo "Generating MMLUProHist test data..."
    python gen_hist_data.py
fi

echo "All files present. Starting evaluations."
echo ""

# --- Helper: start server and wait for it ---
start_server() {
    local model_path=$1
    echo "Starting server with $model_path ..."
    python -m llama_cpp.server --model "$model_path" --port $PORT --n_ctx 2048 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    echo "Waiting for server to be ready..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start within 120 seconds"
    kill $SERVER_PID 2>/dev/null
    exit 1
}

stop_server() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        sleep 2
        echo "Server stopped."
    fi
}

# Cleanup on exit
trap stop_server EXIT

# =============================================
echo "============================================"
echo "PHASE 1: Evaluate BASE model"
echo "============================================"
start_server "$BASE_GGUF"

echo ""
echo "--- Eval 1/4: Base model on MMLUProComp (in-domain baseline) ---"
python test.py --dataset MMLUProComp --model_name base-qwen $TEST_ARGS

echo ""
echo "--- Eval 2/4: Base model on MMLUProHist (domain shift baseline) ---"
python test.py --dataset MMLUProHist --model_name base-qwen $TEST_ARGS

stop_server

# =============================================
echo ""
echo "============================================"
echo "PHASE 2: Evaluate DISTILLED model"
echo "============================================"
start_server "$DISTILLED_GGUF"

echo ""
echo "--- Eval 3/4: Distilled model on MMLUProComp (in-domain) ---"
python test.py --dataset MMLUProComp --model_name distilled-qwen $TEST_ARGS

echo ""
echo "--- Eval 4/4: Distilled model on MMLUProHist (domain shift) ---"
python test.py --dataset MMLUProHist --model_name distilled-qwen $TEST_ARGS

stop_server

# =============================================
echo ""
echo "============================================"
echo "PHASE 3: Error Analysis"
echo "============================================"

# Find the most recent results folders
COMP_BASE=$(ls -td ./tmp_test/MMLUProComp_base-qwen_* 2>/dev/null | head -1)
HIST_BASE=$(ls -td ./tmp_test/MMLUProHist_base-qwen_* 2>/dev/null | head -1)
COMP_DIST=$(ls -td ./tmp_test/MMLUProComp_distilled-qwen_* 2>/dev/null | head -1)
HIST_DIST=$(ls -td ./tmp_test/MMLUProHist_distilled-qwen_* 2>/dev/null | head -1)

echo ""
echo "--- Error Analysis 1/4: Base on MMLUProComp ---"
python error_analysis.py --dataset MMLUProComp --results_dir "$COMP_BASE" --num_examples 3

echo ""
echo "--- Error Analysis 2/4: Base on MMLUProHist ---"
python error_analysis.py --dataset MMLUProHist --results_dir "$HIST_BASE" --num_examples 3

echo ""
echo "--- Error Analysis 3/4: Distilled on MMLUProComp ---"
python error_analysis.py --dataset MMLUProComp --results_dir "$COMP_DIST" --num_examples 3

echo ""
echo "--- Error Analysis 4/4: Distilled on MMLUProHist ---"
python error_analysis.py --dataset MMLUProHist --results_dir "$HIST_DIST" --num_examples 3

echo ""
echo "============================================"
echo "ALL DONE! Summary of result folders:"
echo "============================================"
echo "  Base + CompSci:    $COMP_BASE"
echo "  Base + History:    $HIST_BASE"
echo "  Distilled + CompSci: $COMP_DIST"
echo "  Distilled + History: $HIST_DIST"
