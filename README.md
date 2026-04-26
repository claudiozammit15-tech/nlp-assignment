# Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement

# Introduction
In this work, we introduce a novel **D**ebate and **R**eflect (**D\&R**) framework that combines the strengths of multi-agent debate and actionable teacher feedback to robustly enhance smaller models.

# Code

## Environment
Details of the conda env:
```
python      3.12.7
cuda        12.4.1
pytorch     2.4.1
accelerate  1.5.2
anthropic   0.42.0
datasets    3.1.0
google-generativeai 0.8.3
numpy       2.1.3
openai      1.59.6
peft        0.15.0
transformers    4.50.0
trl         0.16.0
vllm        0.8.2
```

## Usage

>  Some configs are specified in the code, such as the vllm port. Please read all code at first. Then configure the code.

1. Setup your api keys for gpt, claude, and gemini, i.e. environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `ANTHROPIC_API_KEY`. Serve mistral with vllm.
2. Run `get_and_split_data.py` for MMLU Pro. MAGs (without debate) are stored in `./data/MAG` and `./data/test_Data`. For MATH, use the MAGs released in [MAGDi](https://github.com/dinobby/MAGDi) (we don't use the debates in their MAGs).
3. Run `python ./debate_teachers_student_w_critique_test.py --dataset MMLUProPhys`. Debates are recorded in new MAGs in `./data/MAG_new_mistral`.
4. Run `python ./mag2sft.py` and `python ./mag2preference.py` to construct SFT and T-DPO training data.
5. Run `CUDA_VISIBLE_DEVICES="X,X,X,X" bash ./run_sft.sh`. (Training SFT)
6. Run `CUDA_VISIBLE_DEVICES="X,X,X,X" bash ./run_tdpo.sh`. (Training T-DPO)
7. Serve the distilled student model with vllm and run `python ./test.py --dataset MMLUProPhys --model_name hosted_vllm/tdpo --num_proc 200 --max_new_tokens 700 --temperature 0.3 --top_p 0.9`. (Evaluation)

Some code files are based on the code from [MAGDi](https://github.com/dinobby/MAGDi), [Huggingface TRL examples](https://github.com/huggingface/trl), and [MATH](https://github.com/hendrycks/math). Thanks for their code.