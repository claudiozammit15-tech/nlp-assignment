{
    export TOKENIZERS_PARALLELISM=false

    accelerate launch --config_file ./ddp.yaml ./tdpo_after_sft.py \
        --dataset_name ./data/mag_new_preference_pair_w_mistral/MMLUProComp_mag_new_974_preference_pair_w_mistral.jsonl \
        --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
        --learning_rate 5.0e-6 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --max_length 1024 \
        --max_prompt_length 512 \
        --truncation_mode keep_end \
        --no_remove_unused_columns \
        --use_peft \
        --warmup_ratio 0.1 \
        --logging_steps 0.1 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 0.1 \
        --output_dir ./tmp_tdpo/tdpo_after_sft_MMLUProBComp_mag_new_974_w_qwen_3epoch \
        --torch_dtype float32 \
        --dataloader_num_workers 0 \
        --precompute_ref_log_probs \
        --model_adapter_name tdpo \
        --ref_adapter_name reference \
        --sft_adapter_path ./tmp_sft/sft_chosen_MMLUProComp_mag_new_974_w_qwen_2epoch \
        --ddp_find_unused_parameters true \
        --seed 42
        # --lora_r 16 \
        # --lora_alpha 32 \
        # --lora_task_type CAUSAL_LM \
        # --group_by_length
        # --gradient_checkpointing \

    exit
}
