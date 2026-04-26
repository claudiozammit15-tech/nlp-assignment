#!/bin/bash
# =============================================================================
# run_pipeline.sh
# Runs the full D&R pipeline: debate → training data → SFT → T-DPO → merge
#
# BEFORE RUNNING:
#   1. Activate the environment:  conda activate dr_env
#   2. Export your API keys (see README.md)
#   3. Run from the repository root:  bash replications/run_pipeline.sh
#
# NOTE: Training runs on CPU. Expect several hours total.
# NOTE: GGUF conversion (Step 6) requires llama.cpp — set LLAMA_CPP_DIR below.
# =============================================================================

set -e

# ── Set your llama.cpp path here before running ───────────────────────────────
LLAMA_CPP_DIR=""   # e.g. /home/user/llama.cpp

# ── Check API keys are set ────────────────────────────────────────────────────
for KEY in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY; do
    if [[ -z "${!KEY}" ]]; then
        echo "ERROR: $KEY is not set. Please export it before running."
        exit 1
    fi
done

echo "======================================================"
echo " D&R Pipeline"
echo "======================================================"

# ── Step 1: Split data ────────────────────────────────────────────────────────
echo ""
echo "[1/6] Splitting MMLU-Pro Computer Science data..."
python get_and_split_data.py

# ── Step 2: Run debate ────────────────────────────────────────────────────────
echo ""
echo "[2/6] Starting student model server (port 58000)..."
python -m llama_cpp.server \
    --model ./models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 58000 \
    --n_ctx 4096 \
    --n_threads "$(nproc)" \
    > /tmp/llama_debate_server.log 2>&1 &
DEBATE_SERVER_PID=$!

echo "  Waiting for student model server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:58000/v1/models" > /dev/null 2>&1; then
        echo "  Server ready."
        break
    fi
    sleep 5
    if [[ $i -eq 120 ]]; then
        echo "  ERROR: Student model server failed to start. Check /tmp/llama_debate_server.log"
        exit 1
    fi
done

echo "[2/6] Running multi-agent debate (10 questions)..."
python debate_teachers_student_w_critique.py --dataset MMLUProComp

echo "  Stopping student model server..."
kill "$DEBATE_SERVER_PID" 2>/dev/null || true
wait "$DEBATE_SERVER_PID" 2>/dev/null || true
echo "  Server stopped."

# ── Step 3: Build SFT training data ──────────────────────────────────────────
echo ""
echo "[3/6] Building SFT training data..."
python mag2sft.py

# ── Step 4: Build T-DPO preference pairs ─────────────────────────────────────
echo ""
echo "[4/6] Building T-DPO preference pairs..."
python mag2preference.py

# ── Step 5a: SFT training ─────────────────────────────────────────────────────
echo ""
echo "[5a/6] SFT training (CPU — several hours)..."
CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false python sft.py \
    --dataset_name ./data/mag_new_sft_w_mistral/MMLUProComp_mag_new_974_sft_w_mistral.jsonl \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --learning_rate 2.0e-4 \
    --num_train_epochs 2 \
    --packing false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 512 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_task_type CAUSAL_LM \
    --logging_steps 0.1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 0.5 \
    --output_dir ./tmp_sft/sft_chosen_MMLUProComp_mag_new_974_w_qwen_2epoch \
    --optim adafactor \
    --torch_dtype float32 \
    --bf16 false \
    --fp16 false \
    --dataloader_num_workers 1 \
    --seed 42

# ── Step 5b: T-DPO training ───────────────────────────────────────────────────
echo ""
echo "[5b/6] T-DPO training (CPU — several hours)..."
CUDA_VISIBLE_DEVICES="" python tdpo_after_sft.py \
    --dataset_name ./data/mag_new_preference_pair_w_mistral/MMLUProComp_mag_new_974_preference_pair_w_mistral.jsonl \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 512 \
    --max_prompt_length 256 \
    --truncation_mode keep_end \
    --no_remove_unused_columns \
    --use_peft \
    --warmup_ratio 0.1 \
    --logging_steps 0.1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 0.1 \
    --output_dir ./tmp_tdpo/tdpo_after_sft_MMLUProComp_mag_new_974_w_qwen_3epoch \
    --torch_dtype float32 \
    --dataloader_num_workers 0 \
    --precompute_ref_log_probs \
    --model_adapter_name tdpo \
    --ref_adapter_name reference \
    --sft_adapter_path ./tmp_sft/sft_chosen_MMLUProComp_mag_new_974_w_qwen_2epoch \
    --optim adafactor \
    --gradient_checkpointing \
    --seed 42

# ── Step 6: Merge adapter + convert to GGUF ───────────────────────────────────
echo ""
echo "[6/6] Merging LoRA adapter into base model..."
python replications/merge_adapter.py \
    --adapter_path ./tmp_tdpo/tdpo_after_sft_MMLUProComp_mag_new_974_w_qwen_3epoch \
    --base_model   Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir   ./models/qwen-distilled-merged \
    --adapter_name tdpo

echo ""
if [[ -z "$LLAMA_CPP_DIR" ]]; then
    echo "LLAMA_CPP_DIR is not set. Run GGUF conversion manually:"
    echo ""
    echo "  python \$LLAMA_CPP_DIR/convert_hf_to_gguf.py ./models/qwen-distilled-merged \\"
    echo "         --outfile ./models/qwen-distilled.gguf --outtype f16"
else
    echo "Converting to GGUF..."
    python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ./models/qwen-distilled-merged \
        --outfile ./models/qwen-distilled.gguf \
        --outtype f16
fi

echo ""
echo "======================================================"
echo " Pipeline complete!"
echo " Next: bash replications/evaluate.sh"
echo "======================================================"
