"""
merge_adapter.py
----------------
Merges the trained LoRA adapter into the base Qwen2.5-1.5B-Instruct weights,
producing a standalone HuggingFace model directory ready for GGUF conversion.

This is called automatically by run_pipeline.sh. You should not need to run
this manually unless you are re-doing the merge step in isolation.

Usage:
    python replications/merge_adapter.py \
        --adapter_path ./tmp_tdpo/tdpo_after_sft_MMLUProComp_mag_new_974_w_qwen_3epoch \
        --base_model   Qwen/Qwen2.5-1.5B-Instruct \
        --output_dir   ./models/qwen-distilled-merged \
        --adapter_name tdpo
"""

import argparse
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_model",   type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--adapter_name", type=str, default="tdpo")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype="float32", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    adapter_dir = os.path.join(args.adapter_path, args.adapter_name)
    if os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        load_path = adapter_dir
    else:
        load_path = args.adapter_path
    model = PeftModel.from_pretrained(model, load_path, adapter_name=args.adapter_name)

    model.set_adapter(args.adapter_name)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
