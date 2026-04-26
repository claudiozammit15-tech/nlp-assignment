import argparse
import json
import random

from transformers import AutoTokenizer

from templates import prompt_templates, choice_prompts

def main(run_suffix=""):
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    for dataset_name in ["MMLUProComp"]: # 'ARC', 'SQA', 'ECQA', 'GSM8K', 'MATH', , "MMLUProComp", "MMLUProChem", "MMLUProMath", "MMLUProBiol"
        with open("./data/MAG_new_mistral/%s_1000%s.jsonl" % (dataset_name, run_suffix), mode='r', encoding="utf-8") as fmag:
            all_mags=[]
            lines=fmag.readlines()
            for l in lines:
                all_mags.append(json.loads(l))
        
        
        all_mags=all_mags[:600]
        examples=[]
        choice_p=choice_prompts[dataset_name]
        if dataset_name == "MATH":
            choice_p=choice_prompts["MATH_train"]
        for mag in all_mags:
            question=mag["question"]

            for round_idx in range(4):
                if ("right_%d" % round_idx) in mag:
                    # round idx exists.
                    current_round_right=mag["right_%d" % round_idx]
                    chosen_stns=[]
                    rejected_stns=[]
                    for k, v in current_round_right.items():
                        if v:
                            # correct
                            chosen_stns.append(mag["%s_output_%d" % (k, round_idx)])
                        else:
                            # incorrect
                            rejected_stns.append(mag["%s_output_%d" % (k, round_idx)])
                    # pair chosen and rejected after distinguish them.
                    for chosen_ans in chosen_stns:
                        for rejected_ans in rejected_stns:
                            if round_idx == 0:
                                # round 0 no debate
                                example={"prompt": (prompt_templates["train_input_template"] % (choice_p, question)), "chosen": ("### Answer\n%s\nFinal answer: %s" % (chosen_ans["reasoning"], chosen_ans["answer"])), "rejected": ("### Answer\n%s\nFinal answer: %s" % (rejected_ans["reasoning"], rejected_ans["answer"]))}
                                examples.append(example)
                            else:
                                last_pool=[]
                                for model_name in current_round_right.keys():
                                    last_reasoning=mag["%s_output_%d" % (model_name, (round_idx-1))]["reasoning"]
                                    if str(last_reasoning) == "None":
                                        continue
                                    last_answer=mag["%s_output_%d" % (model_name, (round_idx-1))]["answer"]
                                    last_c_pool=mag["critique_pool_%d" % (round_idx-1)]
                                    last_reflection="None"
                                    last_feedback=""
                                    f_id=1
                                    for critique in last_c_pool:
                                        if critique[0] == model_name and critique[1] == model_name:
                                            last_reflection=critique[2]
                                        elif critique[1] == model_name:
                                            last_feedback=last_feedback+("%d. %s\n\n" % (f_id, critique[2]))
                                            f_id+=1
                                    if len(last_feedback) <= 0:
                                        last_feedback="None"
                                    last_round_answer_new="%s\nFinal answer: %s\n\nSelf-reflection: %s\n\nFeedback:\n%s\n\n" % (last_reasoning, last_answer, last_reflection, last_feedback)
                                    # truncate answer to control length.
                                    if len(tokenizer.tokenize(last_round_answer_new)) > 950:
                                        last_reasoning_tokens=tokenizer.tokenize(last_reasoning, add_special_tokens=False)
                                        last_reasoning=tokenizer.convert_tokens_to_string(last_reasoning_tokens[:470])
                                        last_round_answer_new="%s\nFinal answer: %s\n\nSelf-reflection: %s\n\nFeedback:\n%s\n\n" % (last_reasoning, last_answer, last_reflection, last_feedback)
                                    if len(tokenizer.tokenize(last_round_answer_new)) > 950:
                                        tmp_feedback="Self-reflection: %s\n\nFeedback:\n%s\n\n" % (last_reflection, last_feedback)
                                        tmp_feedback_tokens=tokenizer.tokenize(tmp_feedback, add_special_tokens=False)
                                        tmp_feedback=tokenizer.convert_tokens_to_string(tmp_feedback_tokens[:470])
                                        last_round_answer_new="%s\nFinal answer: %s\n\n%s" % (last_reasoning, last_answer, tmp_feedback)
                                    last_pool.append(last_round_answer_new)
                                random.shuffle(last_pool)
                                last_pool=sorted(last_pool, key=lambda s: len(tokenizer.tokenize(s)))

                                last_round_answers=""
                                last_round_answers_len=0
                                a_id=1
                                for last_round_answer_new in last_pool:
                                    if last_round_answers_len+len(tokenizer.tokenize(("Answer %d:\n" % a_id)+last_round_answer_new)) > 1000:
                                        # too long
                                        example={"prompt": (prompt_templates["train_debate_input_template"] % (choice_p, question, last_round_answers)), "chosen": ("### Answer\n%s\nFinal answer: %s" % (chosen_ans["reasoning"], chosen_ans["answer"])), "rejected": ("### Answer\n%s\nFinal answer: %s" % (rejected_ans["reasoning"], rejected_ans["answer"]))}
                                        examples.append(example)
                                        a_id=1
                                        last_round_answers=("Answer %d:\n" % a_id)+last_round_answer_new
                                        last_round_answers_len=len(tokenizer.tokenize(("Answer %d:\n" % a_id)+last_round_answer_new))
                                        a_id=2
                                    else:
                                        # not enough
                                        last_round_answers=last_round_answers+("Answer %d:\n" % a_id)+last_round_answer_new
                                        last_round_answers_len=len(tokenizer.tokenize(last_round_answers))
                                        a_id+=1
                                if len(last_round_answers) <= 0:
                                    last_round_answers="None"
                                example={"prompt": (prompt_templates["train_debate_input_template"] % (choice_p, question, last_round_answers)), "chosen": ("### Answer\n%s\nFinal answer: %s" % (chosen_ans["reasoning"], chosen_ans["answer"])), "rejected": ("### Answer\n%s\nFinal answer: %s" % (rejected_ans["reasoning"], rejected_ans["answer"]))}
                                examples.append(example)

        with open("./data/mag_new_preference_pair_w_mistral/%s_mag_new_974_preference_pair_w_mistral%s.jsonl" % (dataset_name, run_suffix), mode='w', encoding="utf-8") as fout:
            for example in examples:
                fout.write(json.dumps(example,ensure_ascii=False)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_suffix", type=str, default="", help="Suffix for input/output filenames, e.g. '_5q'")
    args = parser.parse_args()
    main(run_suffix=args.run_suffix)
