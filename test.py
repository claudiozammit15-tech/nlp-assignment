import argparse
import datetime
import json
import random

from datasets import Dataset
from openai import OpenAI

import data_utils
from templates import prompt_templates, choice_prompts

random.seed(42)

def parse_answer(response, dataset):
    ret=None
    if "Final answer:" in response:
        response_split=response.split("Final answer:")
        if len(response_split) >=2:
            answer=data_utils.parse_answer(dataset, response_split[1])
            ret=answer
    if not ret:
        ret=data_utils.parse_answer_fallback(dataset, response)

    return ret

def prepare_test_data(dataset):
    test_samples = json.load(open(f"./data/test_data/{dataset}_test.json", "r"))
    choice_p=choice_prompts[dataset]

    test_batch = []
    for i in test_samples: 
        messages=[
            {"role": "user", "content": prompt_templates["test_input_template"] % (choice_p, i["question"])}
        ]
        test_batch.append({"messages": messages, "question": i["question"]})

    return test_samples, test_batch

def main():
    parser = argparse.ArgumentParser()
    # dataset: ['SQA', 'ECQA', 'ARC', 'GSM8K', 'MATH']
    parser.add_argument('--dataset', default='MATH', type=str)
    parser.add_argument('--num_proc', default=10, type=int)
    parser.add_argument('--model_name', default='Mistral-7B-Instruct-v0.2', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_p', default=1.0, type=float)
    parser.add_argument('--max_new_tokens', default=None, type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--port", default=58000, type=int)
    args = parser.parse_args()

    print(f"{datetime.datetime.now().strftime('%y-%m-%d %H:%M')} === evaluating {args.model_name} on {args.dataset} dataset ===")

    generate_conf={}
    if args.model_name.startswith("gpt"):
        generate_conf["max_completion_tokens"]=args.max_new_tokens
    else:
        generate_conf["max_tokens"]=args.max_new_tokens
        generate_conf["max_completion_tokens"]=args.max_new_tokens
    if args.model_name.startswith("hosted_vllm"):
        generate_conf["api_base"]="http://localhost:%d/v1" % args.port
        generate_conf["api_key"]="EMPTY"
    if args.greedy:
        generate_conf["temperature"]=0
    else:
        generate_conf["top_p"]=args.top_p
        generate_conf["temperature"]=args.temperature
    print(generate_conf)

    test_samples, test_batch = prepare_test_data(args.dataset)
    print(len(test_samples), len(test_batch))
    test_dataset=Dataset.from_list(test_batch)

    def get_answer_from_api(example):
        messages=example["messages"]

        client=OpenAI(base_url="http://localhost:%d/v1" % args.port, api_key="EMPTY")
        response=client.chat.completions.create(model=args.model_name, messages=messages, max_tokens=generate_conf["max_tokens"], temperature=generate_conf["temperature"], top_p=generate_conf["top_p"])
        answer=parse_answer(response.choices[0].message.content, args.dataset)

        return {"answer": answer, "response": response.choices[0].message.content, "total_tokens": response.usage.total_tokens, "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}

    answered_dateset=test_dataset.map(get_answer_from_api, load_from_cache_file=False, num_proc=args.num_proc)

    pred_ans = answered_dateset["answer"]
    print(len(pred_ans))
    print("token cnt: ", sum(answered_dateset["total_tokens"])/len(answered_dateset["total_tokens"]))
    assert len(test_samples) == len(pred_ans)
    acc = data_utils.calc_acc(args.dataset, pred_ans, test_samples)
    print(f"{datetime.datetime.now().strftime('%y-%m-%d %H:%M')} | samples evaluated: {len(answered_dateset)} | accuracy: {acc}")

    answered_dateset.save_to_disk("./tmp_test/%s_%s_%s" % (args.dataset, args.model_name.replace("/", "_"), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

if __name__ == '__main__':
    main()
