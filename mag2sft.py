import argparse
import json

from templates import prompt_templates, choice_prompts

import data_utils

def main(run_suffix=""):
    for dataset_name in ["MMLUProComp"]: # 'ARC', 'SQA', 'ECQA', 'GSM8K', 'MATH', , "MMLUProComp", "MMLUProChem", "MMLUProMath", "MMLUProBiol"
        with open("./data/MAG_new_mistral/%s_1000%s.jsonl" % (dataset_name, run_suffix), mode='r', encoding="utf-8") as f:
            all_mags = [json.loads(line) for line in f.readlines()]

        choice_p=choice_prompts[dataset_name]
        if dataset_name == "MATH":
            choice_p=choice_prompts["MATH_train"]
        examples=[]
        for mag in all_mags:
            question=mag["question"]
            prompt=prompt_templates["train_input_template"] % (choice_p, question)
            chosen_stns=[]
            for round_idx in range(4):
                if ("gpt-4o_output_%d" % round_idx) in mag:
                    for m_name in ["gpt-4o", "claude-3-5", "gemini-1-5"]:
                        if data_utils.is_equiv(str(mag["%s_output_%d" % (m_name, round_idx)]["answer"]).strip('$'), mag["gold_answer"]):
                            chosen_stns.append(mag["%s_output_%d" % (m_name, round_idx)])

            for chosen_stn in chosen_stns:
                example={"prompt": prompt, "completion": ("### Answer\n%s\nFinal answer: %s" % (chosen_stn["reasoning"], '$'+chosen_stn["answer"].strip('$')+'$'))}
                examples.append(example)

        with open("./data/mag_new_sft_w_mistral/%s_mag_new_974_sft_w_mistral%s.jsonl" % (dataset_name, run_suffix), mode='w', encoding="utf-8") as fout:
            for example in examples:
                fout.write(json.dumps(example,ensure_ascii=False)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_suffix", type=str, default="", help="Suffix for input/output filenames, e.g. '_5q'")
    args = parser.parse_args()
    main(run_suffix=args.run_suffix)
