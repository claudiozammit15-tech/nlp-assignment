set -e

PORT=58000

BASE_GGUF="./models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
BASE_NAME="qwen2.5-1.5b-instruct-q4_k_m"

DISTILLED_GGUF="./models/qwen-distilled.gguf"
DISTILLED_NAME="distilled-qwen"

SERVER_PID=""

start_server() {
    echo "  Starting server: $1"
    python -m llama_cpp.server \
        --model "$1" \
        --host 0.0.0.0 \
        --port $PORT \
        --n_ctx 4096 \
        --n_threads "$(nproc)" \
        > /tmp/llama_server.log 2>&1 &
    SERVER_PID=$!

    echo "  Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "  Server ready."
            return
        fi
        sleep 5
    done
    echo "  ERROR: Server failed to start after 10 minutes. Check /tmp/llama_server.log"
    exit 1
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        echo "  Server stopped."
    fi
}

trap stop_server EXIT

# ── Evaluation of the base model
echo " Evaluating base model"
echo "======================================================"
start_server "$BASE_GGUF"

python test.py \
    --dataset MMLUProComp \
    --model_name "$BASE_NAME" \
    --num_proc 1 \
    --max_new_tokens 700 \
    --temperature 0.3 \
    --top_p 0.9 \
    --port $PORT

stop_server

# ── Evaluation of the distilled model
echo ""
echo " Evaluating DISTILLED model"
echo "======================================================"
start_server "$DISTILLED_GGUF"

python test.py \
    --dataset MMLUProComp \
    --model_name "$DISTILLED_NAME" \
    --num_proc 1 \
    --max_new_tokens 700 \
    --temperature 0.3 \
    --top_p 0.9 \
    --port $PORT

stop_server

echo " Evaluation complete. Results saved to ./tmp_test/"
echo "======================================================"
