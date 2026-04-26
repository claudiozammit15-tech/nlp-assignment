{
    export TOKENIZERS_PARALLELISM=false

    accelerate launch --config_file ./ddp.yaml ./sft.py \
        --dataset_name ./data/mag_new_sft_w_mistral/MMLUProComp_mag_new_974_sft_w_mistral.jsonl \
        --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
        --learning_rate 2.0e-4 \
        --num_train_epochs 2 \
        --packing false \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --max_seq_length 1024 \
        --use_peft \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_task_type CAUSAL_LM \
        --logging_steps 0.1 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 0.5 \
        --output_dir ./tmp_sft/sft_chosen_MMLUProComp_mag_new_974_w_qwen_2epoch \
        --torch_dtype float32 \
        --dataloader_num_workers 1 \
        --seed 42
        # --lora_target_modules "q_proj" "v_proj" \
        # --ddp_find_unused_parameters false \
        # --gradient_checkpointing \
        # --gradient_checkpointing_use_reentrant true \

    exit
}