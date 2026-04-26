# D&R Replication Guide

Reproduction package for the Debate & Reflect (D&R) experiment on MMLU-Pro Computer Science.

---

## Requirements

- Windows 10/11 with WSL2 (Ubuntu)
- 16 GB RAM minimum (32 GB recommended for training)
- No GPU required — everything runs on CPU
- API keys for: OpenAI, Anthropic, Google Gemini

---

## Step 0 — Open WSL and navigate to the repo

```bash
wsl -d Ubuntu
cd /mnt/c/Users/<your-username>/Downloads/D-R-master/D-R-master
```

---

## Step 1 — Set up the environment (run once)

```bash
chmod +x replications/setup_environment.sh
bash replications/setup_environment.sh
conda activate dr_env
```

---

## Step 2 — Set your API keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
```

To make these permanent, add them to `~/.bashrc`.

---

## Step 3 — (Optional) Set your llama.cpp path

GGUF conversion at the end of the pipeline requires [llama.cpp](https://github.com/ggerganov/llama.cpp).
Open `replications/run_pipeline.sh` and set the `LLAMA_CPP_DIR` variable at the top to your llama.cpp install path.

If you skip this, the script will print the manual conversion command for you to run separately.

---

## Step 4 — Run the full pipeline

```bash
conda activate dr_env
bash replications/run_pipeline.sh
```

This runs in order:
1. Download and split MMLU-Pro CS data (411 questions → 50/50 train/test)
2. Run multi-agent debate on 10 training questions (GPT-4o, Claude, Gemini, Qwen)
3. Build SFT training data from debate output
4. Build T-DPO preference pairs from debate output
5. Train SFT adapter (2 epochs, CPU, ~2–4 hours)
6. Train T-DPO adapter (3 epochs, CPU, ~3–6 hours)
7. Merge LoRA adapter into base model
8. Convert merged model to GGUF format

**Note:** If you re-run this from scratch, delete the existing debate output first to avoid appending to it:
```bash
rm -f data/MAG_new_mistral/MMLUProComp_1000.jsonl
```

---

## Step 5 — Evaluate both models

```bash
bash replications/evaluate.sh
```

This starts the inference server, evaluates each model against the full test set, then stops the server. Each model takes approximately 1–2 hours on CPU.

Results are saved to `./tmp_test/`.

**Results obtained in this experiment (2026-04-26):**

| Model | Correct | Total | Accuracy |
|---|---|---|---|
| Base (Qwen2.5-1.5B-Instruct Q4) | 24 | 137 | **17.52%** |
| Distilled (after 10-question D&R) | 31 | 137 | **22.63%** |

The distilled model outperforms the base by +5.1 percentage points. Note that results may vary slightly between runs due to temperature=0.3 sampling over a 137-question test set.

---

## Notes on deviations from the original paper

| | Original paper | This implementation |
|---|---|---|
| Student model | Mistral-7B-Instruct | Qwen2.5-1.5B-Instruct (7B too large for CPU) |
| Teacher 2 | Claude 3.5 Sonnet | Claude Opus 4.6 |
| Teacher 3 | Gemini 1.5 Flash | Gemini 2.5 Flash |
| Inference | vLLM (GPU) | llama-cpp-python GGUF (CPU) |
| Debate scale | Full training set | 10 questions (CPU time constraint) |
| Sequence length | 1024 / 512 tokens | 512 / 256 tokens (CPU RAM constraint) |

The absolute accuracy is limited by the 10-question training set. Full-scale training (~274 questions) would require an estimated 75–105 hours on CPU.
