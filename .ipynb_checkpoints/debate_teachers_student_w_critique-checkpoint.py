import argparse
import datetime
import json
import logging
import os
import random
import time

from anthropic import Anthropic
from openai import OpenAI

from batch_api import openai_batch, claude_batch, gemini_batch, vllm_batch, gemini_batch_feedback, vllm_batch_feedback
import data_utils
from templates import prompt_templates, choice_prompts
from utils import parse_answer, parse_feedback

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler=logging.StreamHandler()
formatter=logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(lineno)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
random.seed(42)

gen_conf={
    "claude-3-5": {"model": "claude-opus-4-6", "temperature": 0.3, "max_tokens": 1000},
    "gpt-4o": {"model": "gpt-4o-2024-08-06", "temperature": 0.3, "max_completion_tokens": 1000},
    "gemini-1-5": {"model": "gemini-2.5-flash", "temperature": 0.3, "max_tokens": 1000, "api_key": os.environ.get("GEMINI_API_KEY"), "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "max_retries": 3, "num_retries": 3, "cooldown_time": 20},
    "mistral-7b": {"model": "qwen2.5-1.5b-instruct-q4_k_m", "temperature": 0.3, "max_tokens": 512, "n": 1}
}

model_list=["gpt-4o", "claude-3-5", "gemini-1-5", "mistral-7b"]

token_cnt={
    "claude-3-5": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
    "gpt-4o": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
    "gemini-1-5": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
    "mistral-7b": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
}

def right_or_wrong(answer, gold, dataset):
    if dataset in ["SQA", "ECQA", "ARC", "MMLUProPhys", "MMLUProComp", "MMLUProChem", "MMLUProMath", "MMLUProBiol", "MMLUProHist"]:
        right=answer == gold
    elif dataset == "GSM8K":
        right=float(answer) == float(gold)
    elif dataset == "MATH":
        right=data_utils.is_equiv(str(answer).strip('$'), gold)
    return right

def main(args):
    openai_client=OpenAI()
    claude_client=Anthropic()
    world_size=1
    num_proc=1
    now_time=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def round_end(mags, round_idx):
        messages_batch={model_name: [] for model_name in model_list}
        cnt=0
        for mag in mags:
            if not mag["stop"]:
                question=mag["question"]
                gold=mag["gold_answer"]
                mag["critique_pool_%d" % round_idx]=[]

                for model_name in model_list:
                    if mag["right_%d" % round_idx][model_name]:
                        # It is correct and provides critiques to others.
                        right_reasoning=mag["%s_output_%d" % (model_name, round_idx)]["reasoning"]
                        right_answer=mag["%s_output_%d" % (model_name, round_idx)]["answer"]
                        for model_name_wrong in [k for k,v in mag["right_%d" % round_idx].items() if not v]:
                            wrong_reasoning=mag["%s_output_%d" % (model_name_wrong, round_idx)]["reasoning"]
                            wrong_answer=mag["%s_output_%d" % (model_name_wrong, round_idx)]["answer"]
                            messages=[{"role": "user", "content": prompt_templates["critic_template"] % (question, right_reasoning, right_answer, wrong_reasoning, wrong_answer)}]
                            messages_batch[model_name].append(("%d_%s_%s" % (mag["id"], model_name, model_name_wrong), messages))
                            cnt+=1
                    else:
                        # It is incorrect and self-reflects.
                        reasoning=mag["%s_output_%d" % (model_name, round_idx)]["reasoning"]
                        answer=mag["%s_output_%d" % (model_name, round_idx)]["answer"]
                        messages=[{"role": "user", "content": prompt_templates["self_reflection_template"] % (question, '$'+gold+'$', reasoning, answer)}]
                        messages_batch[model_name].append(("%d_%s_%s" % (mag["id"], model_name, model_name), messages))
                        cnt+=1

        if cnt <= 0:
            return

        openai_batch_id=openai_batch(openai_client, messages_batch["gpt-4o"], args.dataset, round_idx, operation="input_feedback", time=now_time, gen_config=gen_conf["gpt-4o"])
        claude_batch_id=claude_batch(claude_client, messages_batch["claude-3-5"], gen_conf["claude-3-5"])

        gemini_answered_dataset=gemini_batch_feedback(messages_batch["gemini-1-5"], gen_conf["gemini-1-5"], num_proc)
        for gemini_answer in gemini_answered_dataset:
            mag_id, model_name_from, model_name_to=gemini_answer["id"].split('_')
            mag_id=int(mag_id)
            tgt_mag_idx=None
            for i in range(len(mags)):
                if int(mags[i]["id"]) == mag_id:
                    tgt_mag_idx=i
            mags[tgt_mag_idx]["critique_pool_%d" % round_idx].append((model_name_from, model_name_to, gemini_answer["feedback"], gemini_answer["raw"]))
            token_cnt["gemini-1-5"]["prompt_tokens"]=token_cnt["gemini-1-5"]["prompt_tokens"]+gemini_answer["prompt_tokens"]
            token_cnt["gemini-1-5"]["completion_tokens"]=token_cnt["gemini-1-5"]["completion_tokens"]+gemini_answer["completion_tokens"]
            token_cnt["gemini-1-5"]["cached_tokens"]=token_cnt["gemini-1-5"]["cached_tokens"]+gemini_answer["cached_tokens"]

        mistral_answered_dataset=vllm_batch_feedback(messages_batch["mistral-7b"], gen_conf["mistral-7b"], num_proc, world_size)
        for mistral_answer in mistral_answered_dataset:
            mag_id, model_name_from, model_name_to=mistral_answer["id"].split('_')
            mag_id=int(mag_id)
            tgt_mag_idx=None
            for i in range(len(mags)):
                if int(mags[i]["id"]) == mag_id:
                    tgt_mag_idx=i
            mags[tgt_mag_idx]["critique_pool_%d" % round_idx].append((model_name_from, model_name_to, mistral_answer["feedback"], mistral_answer["raw"]))
            token_cnt["mistral-7b"]["prompt_tokens"]=token_cnt["mistral-7b"]["prompt_tokens"]+mistral_answer["prompt_tokens"]
            token_cnt["mistral-7b"]["completion_tokens"]=token_cnt["mistral-7b"]["completion_tokens"]+mistral_answer["completion_tokens"]
            token_cnt["mistral-7b"]["cached_tokens"]=token_cnt["mistral-7b"]["cached_tokens"]+mistral_answer["cached_tokens"]

        logger.info("sleeping")
        batch_completed=False
        while not batch_completed:
            batch=claude_client.messages.batches.retrieve(claude_batch_id)
            if batch.processing_status in ["in_progress"]:
                time.sleep(20)
            elif batch.processing_status in ["ended"]:
                batch_completed=True
            elif batch.processing_status in ["cancelling"]:
                raise NotImplementedError
        logger.info("claude wakes up")
        batch_done=claude_client.messages.batches.retrieve(claude_batch_id)
        if batch_done.request_counts.errored > 0 or batch_done.request_counts.canceled > 0:
            raise NotImplementedError
        tmp_results=claude_client.messages.batches.results(batch_done.id)
        for res in tmp_results:
            tmp_feedback, tmp_raw=parse_feedback([{"message": {"content": res.result.message.content[0].text}}])
            mag_id, model_name_from, model_name_to=res.custom_id.split('_')
            mag_id=int(mag_id)
            tgt_mag_idx=None
            for i in range(len(mags)):
                if int(mags[i]["id"]) == mag_id:
                    tgt_mag_idx=i
            mags[tgt_mag_idx]["critique_pool_%d" % round_idx].append((model_name_from, model_name_to, tmp_feedback, tmp_raw))
            token_cnt["claude-3-5"]["prompt_tokens"]=token_cnt["claude-3-5"]["prompt_tokens"]+res.result.message.usage.input_tokens
            token_cnt["claude-3-5"]["completion_tokens"]=token_cnt["claude-3-5"]["completion_tokens"]+res.result.message.usage.output_tokens

        batch_completed=False
        while not batch_completed:
            batch=openai_client.batches.retrieve(openai_batch_id)
            if batch.status in ["validating","in_progress", "finalizing"]:
                time.sleep(10)
            elif batch.status in ["completed"]:
                batch_completed=True
            elif batch.status in ["faild", "expired", "cancelling", "cancelled"]:
                raise NotImplementedError
        logger.info("openai wakes up")
        batch_done=openai_client.batches.retrieve(openai_batch_id)
        if batch_done.request_counts.failed > 0:
            raise NotImplementedError
        file_response=openai_client.files.content(batch_done.output_file_id)
        with open("./tmp_batch/zxf_%s_%s_r%d_%s.jsonl" % (args.dataset, now_time, round_idx, "output_feedback"), mode='w', encoding="utf-8") as f_tmp:
            f_tmp.write(file_response.text)
        for line in file_response.text.splitlines():
            if len(line) > 0:
                tmp_response=json.loads(line)
                tmp_feedback, tmp_raw=parse_feedback(tmp_response["response"]["body"]["choices"])
                mag_id, model_name_from, model_name_to=tmp_response["custom_id"].split('_')
                mag_id=int(mag_id)
                tgt_mag_idx=None
                for i in range(len(mags)):
                    if int(mags[i]["id"]) == mag_id:
                        tgt_mag_idx=i
                mags[tgt_mag_idx]["critique_pool_%d" % round_idx].append((model_name_from, model_name_to, tmp_feedback, tmp_raw))
                token_cnt["gpt-4o"]["prompt_tokens"]=token_cnt["gpt-4o"]["prompt_tokens"]+tmp_response["response"]["body"]["usage"]["prompt_tokens"]
                token_cnt["gpt-4o"]["completion_tokens"]=token_cnt["gpt-4o"]["completion_tokens"]+tmp_response["response"]["body"]["usage"]["completion_tokens"]
                if tmp_response["response"]["body"]["usage"]["prompt_tokens_details"]:
                    cached_tokens=tmp_response["response"]["body"]["usage"]["prompt_tokens_details"]["cached_tokens"]
                else:
                    cached_tokens=0
                token_cnt["gpt-4o"]["cached_tokens"]=token_cnt["gpt-4o"]["cached_tokens"]+cached_tokens

    with open("./data/MAG/%s_1000.json" % args.dataset, mode='r') as fmags:
        mags=json.load(fmags)

    if args.dataset in choice_prompts:
        choice_prompt=choice_prompts[args.dataset]
    else:
        raise ValueError("Unkown dataset %s" % args.dataset)

    new_mags=[]
    for mag_id, mag in enumerate(mags):
        new_mag={"id": mag_id}

        question=mag["question"]
        new_mag["question"]=question
        gold=mag["gold_answer"]
        new_mag["gold_answer"]=gold

        new_mags.append(new_mag)

    if args.num_questions is not None:
        new_mags = new_mags[:args.num_questions]
    logger.info("process %d mags.", len(new_mags))

    # Four agents answer questions initially.
    logger.info("round 0")
    messages_batch=[]
    for new_mag in new_mags:
        r0_input=prompt_templates["r0_input_template"] % (choice_prompt, new_mag["question"])
        messages=[{"role": "user", "content": r0_input}]
        messages_batch.append((new_mag["id"], messages))

    # Create two batches for gpt and claude. While waiting them, run gemini and mistral.
    openai_batch_id=openai_batch(openai_client, messages_batch, args.dataset, round_idx=0, operation="input", time=now_time, gen_config=gen_conf["gpt-4o"])
    claude_batch_id=claude_batch(claude_client, messages_batch, gen_conf["claude-3-5"])

    gemini_answered_dataset=gemini_batch(messages_batch, args.dataset, gen_conf["gemini-1-5"], num_proc)
    # build a map from mag ids to responses.
    gemini_answers_map={e["id"]: e for e in gemini_answered_dataset}
    for new_mag in new_mags:
        gemini_answer=gemini_answers_map[new_mag["id"]]
        new_mag["%s_raw_0" % "gemini-1-5"]=gemini_answer["raw"] # raw only for logging
        new_mag["%s_output_0" % "gemini-1-5"]={"reasoning": gemini_answer["reasoning"], "answer": gemini_answer["answer"]}
        token_cnt["gemini-1-5"]["prompt_tokens"]=token_cnt["gemini-1-5"]["prompt_tokens"]+gemini_answer["prompt_tokens"]
        token_cnt["gemini-1-5"]["completion_tokens"]=token_cnt["gemini-1-5"]["completion_tokens"]+gemini_answer["completion_tokens"]
        token_cnt["gemini-1-5"]["cached_tokens"]=token_cnt["gemini-1-5"]["cached_tokens"]+gemini_answer["cached_tokens"]

    mistral_answered_dataset=vllm_batch(messages_batch, args.dataset, gen_conf["mistral-7b"], num_proc, world_size)
    mistral_answers_map={e["id"]: e for e in mistral_answered_dataset}
    for new_mag in new_mags:
        mistral_answer=mistral_answers_map[new_mag["id"]]
        new_mag["%s_raw_0" % "mistral-7b"]=mistral_answer["raw"] # raw only for logging
        new_mag["%s_output_0" % "mistral-7b"]={"reasoning": mistral_answer["reasoning"], "answer": mistral_answer["answer"]}
        token_cnt["mistral-7b"]["prompt_tokens"]=token_cnt["mistral-7b"]["prompt_tokens"]+mistral_answer["prompt_tokens"]
        token_cnt["mistral-7b"]["completion_tokens"]=token_cnt["mistral-7b"]["completion_tokens"]+mistral_answer["completion_tokens"]
        token_cnt["mistral-7b"]["cached_tokens"]=token_cnt["mistral-7b"]["cached_tokens"]+mistral_answer["cached_tokens"]

    # collect two batches
    logger.info("sleeping")
    batch_completed=False
    while not batch_completed:
        batch=claude_client.messages.batches.retrieve(claude_batch_id)
        if batch.processing_status in ["in_progress"]:
            time.sleep(20)
        elif batch.processing_status in ["ended"]:
            batch_completed=True
        elif batch.processing_status in ["cancelling"]:
            raise NotImplementedError
    logger.info("claude wakes up")
    # collect claude's results
    batch_done=claude_client.messages.batches.retrieve(claude_batch_id)
    if batch_done.request_counts.errored > 0 or batch_done.request_counts.canceled > 0:
        raise NotImplementedError
    tmp_results=claude_client.messages.batches.results(batch_done.id)
    claude_answers_map={}
    for res in tmp_results:
        tmp_ans, tmp_raw=parse_answer([{"message": {"content": res.result.message.content[0].text}}], args.dataset)
        claude_answers_map[int(res.custom_id)]={"id": int(res.custom_id), "answer": tmp_ans["answer"], "reasoning": tmp_ans["reasoning"], "raw": tmp_raw, "prompt_tokens":res.result.message.usage.input_tokens, "completion_tokens": res.result.message.usage.output_tokens, "cached_tokens": 0}
    for new_mag in new_mags:
        claude_answer=claude_answers_map[new_mag["id"]]
        new_mag["%s_raw_0" % "claude-3-5"]=claude_answer["raw"] # raw only for logging
        new_mag["%s_output_0" % "claude-3-5"]={"reasoning": claude_answer["reasoning"], "answer": claude_answer["answer"]}
        token_cnt["claude-3-5"]["prompt_tokens"]=token_cnt["claude-3-5"]["prompt_tokens"]+claude_answer["prompt_tokens"]
        token_cnt["claude-3-5"]["completion_tokens"]=token_cnt["claude-3-5"]["completion_tokens"]+claude_answer["completion_tokens"]
        token_cnt["claude-3-5"]["cached_tokens"]=token_cnt["claude-3-5"]["cached_tokens"]+claude_answer["cached_tokens"]

    batch_completed=False
    while not batch_completed:
        batch=openai_client.batches.retrieve(openai_batch_id)
        if batch.status in ["validating","in_progress", "finalizing"]:
            time.sleep(10)
        elif batch.status in ["completed"]:
            batch_completed=True
        elif batch.status in ["faild", "expired", "cancelling", "cancelled"]:
            raise NotImplementedError
    logger.info("openai wakes up")
    # collect gpt's results
    batch_done=openai_client.batches.retrieve(openai_batch_id)
    if batch_done.request_counts.failed > 0:
        raise NotImplementedError
    file_response=openai_client.files.content(batch_done.output_file_id)
    with open("./tmp_batch/zxf_%s_%s_r%d_%s.jsonl" % (args.dataset, now_time, 0, "output"), mode='w', encoding="utf-8") as f_tmp:
        f_tmp.write(file_response.text)
    openai_answers_map={}
    for line in file_response.text.splitlines():
        if len(line) > 0:
            tmp_response=json.loads(line)
            tmp_ans, tmp_raw=parse_answer(tmp_response["response"]["body"]["choices"], args.dataset)
            if tmp_response["response"]["body"]["usage"]["prompt_tokens_details"]:
                cached_tokens=tmp_response["response"]["body"]["usage"]["prompt_tokens_details"]["cached_tokens"]
            else:
                cached_tokens=0
            openai_answers_map[int(tmp_response["custom_id"])]={"id": int(tmp_response["custom_id"]), "answer": tmp_ans["answer"], "reasoning": tmp_ans["reasoning"], "raw": tmp_raw, "prompt_tokens": tmp_response["response"]["body"]["usage"]["prompt_tokens"], "completion_tokens": tmp_response["response"]["body"]["usage"]["completion_tokens"], "cached_tokens": cached_tokens}
    for new_mag in new_mags:
        openai_answer=openai_answers_map[new_mag["id"]]
        new_mag["%s_raw_0" % "gpt-4o"]=openai_answer["raw"] # raw only for logging
        new_mag["%s_output_0" % "gpt-4o"]={"reasoning": openai_answer["reasoning"], "answer": openai_answer["answer"]}
        token_cnt["gpt-4o"]["prompt_tokens"]=token_cnt["gpt-4o"]["prompt_tokens"]+openai_answer["prompt_tokens"]
        token_cnt["gpt-4o"]["completion_tokens"]=token_cnt["gpt-4o"]["completion_tokens"]+openai_answer["completion_tokens"]
        token_cnt["gpt-4o"]["cached_tokens"]=token_cnt["gpt-4o"]["cached_tokens"]+openai_answer["cached_tokens"]

    all_stop=True
    for new_mag in new_mags:
        current_round_right={}
        for model_name in model_list:
            current_round_right[model_name]=right_or_wrong(new_mag["%s_output_0" % (model_name)]["answer"], new_mag["gold_answer"], args.dataset)
        new_mag["right_0"]=current_round_right
        if all(current_round_right.values()):
            new_mag["stop"]=True
        else:
            new_mag["stop"]=False
            all_stop=False
        
    logger.info("round 0.5")
    round_end(new_mags, round_idx=0)

    logger.info("--------------------------------------------------------")

    round_idx=1
    if all_stop:
        round_idx=10

    while round_idx <= 3:
        logger.info("round %d", round_idx)
        messages_batch=[]
        for new_mag in new_mags:
            if new_mag["stop"]:
                continue
            question=new_mag["question"]
            last_round_answers=""
            model_order=[0,1,2,3]
            random.shuffle(model_order)
            for i, model_idx in enumerate(model_order):
                last_reasoning=new_mag["%s_output_%d" % (model_list[model_idx], (round_idx-1))]["reasoning"]
                last_answer=new_mag["%s_output_%d" % (model_list[model_idx], (round_idx-1))]["answer"]
                last_c_pool=new_mag["critique_pool_%d" % (round_idx-1)]
                last_reflection="None"
                last_feedback=""
                f_id=1
                for critique in last_c_pool:
                    if critique[0] == model_list[model_idx] and critique[1] == model_list[model_idx]:
                        last_reflection=critique[2]
                    elif critique[1] == model_list[model_idx]:
                        last_feedback=last_feedback+("%d. %s\n\n" % (f_id, critique[2]))
                        f_id+=1
                if len(last_feedback) <= 0:
                    last_feedback="None"
                last_round_answers=last_round_answers+("Answer %d:\n%s\nFinal answer: %s\n\nSelf-reflection: %s\n\nFeedback:\n%s\n\n" % (i+1, last_reasoning, last_answer, last_reflection, last_feedback))
            if len(last_round_answers) <= 0:
                last_round_answers="None"
            messages=[{"role": "user", "content": prompt_templates["debate_input_template"] % (choice_prompt, question, last_round_answers)}]
            new_mag["debate_last_round_%d" % round_idx]=last_round_answers
            messages_batch.append((new_mag["id"], messages))

        openai_batch_id=openai_batch(openai_client, messages_batch, args.dataset, round_idx, operation="input", time=now_time, gen_config=gen_conf["gpt-4o"])
        claude_batch_id=claude_batch(claude_client, messages_batch, gen_conf["claude-3-5"])

        gemini_answered_dataset=gemini_batch(messages_batch, args.dataset, gen_conf["gemini-1-5"], num_proc)
        gemini_answers_map={e["id"]: e for e in gemini_answered_dataset}
        for new_mag in new_mags:
            if new_mag["stop"]:
                continue
            gemini_answer=gemini_answers_map[new_mag["id"]]
            new_mag["%s_raw_%d" % ("gemini-1-5", round_idx)]=gemini_answer["raw"] # raw only for logging
            new_mag["%s_output_%d" % ("gemini-1-5", round_idx)]={"reasoning": gemini_answer["reasoning"], "answer": gemini_answer["answer"]}
            token_cnt["gemini-1-5"]["prompt_tokens"]=token_cnt["gemini-1-5"]["prompt_tokens"]+gemini_answer["prompt_tokens"]
            token_cnt["gemini-1-5"]["completion_tokens"]=token_cnt["gemini-1-5"]["completion_tokens"]+gemini_answer["completion_tokens"]
            token_cnt["gemini-1-5"]["cached_tokens"]=token_cnt["gemini-1-5"]["cached_tokens"]+gemini_answer["cached_tokens"]

        mistral_answered_dataset=vllm_batch(messages_batch, args.dataset, gen_conf["mistral-7b"], num_proc, world_size)
        mistral_answers_map={e["id"]: e for e in mistral_answered_dataset}
        for new_mag in new_mags:
            if new_mag["stop"]:
                continue
            mistral_answer=mistral_answers_map[new_mag["id"]]
            new_mag["%s_raw_%d" % ("mistral-7b", round_idx)]=mistral_answer["raw"] # raw only for logging
            new_mag["%s_output_%d" % ("mistral-7b", round_idx)]={"reasoning": mistral_answer["reasoning"], "answer": mistral_answer["answer"]}
            token_cnt["mistral-7b"]["prompt_tokens"]=token_cnt["mistral-7b"]["prompt_tokens"]+mistral_answer["prompt_tokens"]
            token_cnt["mistral-7b"]["completion_tokens"]=token_cnt["mistral-7b"]["completion_tokens"]+mistral_answer["completion_tokens"]
            token_cnt["mistral-7b"]["cached_tokens"]=token_cnt["mistral-7b"]["cached_tokens"]+mistral_answer["cached_tokens"]

        logger.info("sleeping")
        batch_completed=False
        while not batch_completed:
            batch=claude_client.messages.batches.retrieve(claude_batch_id)
            if batch.processing_status in ["in_progress"]:
                time.sleep(20)
            elif batch.processing_status in ["ended"]:
                batch_completed=True
            elif batch.processing_status in ["cancelling"]:
                raise NotImplementedError
        logger.info("claude wakes up")
        batch_done=claude_client.messages.batches.retrieve(claude_batch_id)
        if batch_done.request_counts.errored > 0 or batch_done.request_counts.canceled > 0:
            raise NotImplementedError
        tmp_results=claude_client.messages.batches.results(batch_done.id)
        claude_answers_map={}
        for res in tmp_results:
            tmp_ans, tmp_raw=parse_answer([{"message": {"content": res.result.message.content[0].text}}], args.dataset)
            claude_answers_map[int(res.custom_id)]={"id": int(res.custom_id), "answer": tmp_ans["answer"], "reasoning": tmp_ans["reasoning"], "raw": tmp_raw, "prompt_tokens":res.result.message.usage.input_tokens, "completion_tokens": res.result.message.usage.output_tokens, "cached_tokens": 0}
        for new_mag in new_mags:
            if new_mag["stop"]:
                continue
            claude_answer=claude_answers_map[new_mag["id"]]
            new_mag["%s_raw_%d" % ("claude-3-5", round_idx)]=claude_answer["raw"] # raw only for logging
            new_mag["%s_output_%d" % ("claude-3-5", round_idx)]={"reasoning": claude_answer["reasoning"], "answer": claude_answer["answer"]}
            token_cnt["claude-3-5"]["prompt_tokens"]=token_cnt["claude-3-5"]["prompt_tokens"]+claude_answer["prompt_tokens"]
            token_cnt["claude-3-5"]["completion_tokens"]=token_cnt["claude-3-5"]["completion_tokens"]+claude_answer["completion_tokens"]
            token_cnt["claude-3-5"]["cached_tokens"]=token_cnt["claude-3-5"]["cached_tokens"]+claude_answer["cached_tokens"]

        batch_completed=False
        while not batch_completed:
            batch=openai_client.batches.retrieve(openai_batch_id)
            if batch.status in ["validating","in_progress", "finalizing"]:
                time.sleep(10)
            elif batch.status in ["completed"]:
                batch_completed=True
            elif batch.status in ["faild", "expired", "cancelling", "cancelled"]:
                raise NotImplementedError
        logger.info("openai wakes up")
        batch_done=openai_client.batches.retrieve(openai_batch_id)
        if batch_done.request_counts.failed > 0:
            raise NotImplementedError
        file_response=openai_client.files.content(batch_done.output_file_id)
        with open("./tmp_batch/zxf_%s_%s_r%d_%s.jsonl" % (args.dataset, now_time, round_idx, "output"), mode='w', encoding="utf-8") as f_tmp:
            f_tmp.write(file_response.text)
        openai_answers_map={}
        for line in file_response.text.splitlines():
            if len(line) > 0:
                tmp_response=json.loads(line)
                tmp_ans, tmp_raw=parse_answer(tmp_response["response"]["body"]["choices"], args.dataset)
                if tmp_response["response"]["body"]["usage"]["prompt_tokens_details"]:
                    cached_tokens=tmp_response["response"]["body"]["usage"]["prompt_tokens_details"]["cached_tokens"]
                else:
                    cached_tokens=0
                openai_answers_map[int(tmp_response["custom_id"])]={"id": int(tmp_response["custom_id"]), "answer": tmp_ans["answer"], "reasoning": tmp_ans["reasoning"], "raw": tmp_raw, "prompt_tokens": tmp_response["response"]["body"]["usage"]["prompt_tokens"], "completion_tokens": tmp_response["response"]["body"]["usage"]["completion_tokens"], "cached_tokens": cached_tokens}
        for new_mag in new_mags:
            if new_mag["stop"]:
                continue
            openai_answer=openai_answers_map[new_mag["id"]]
            new_mag["%s_raw_%d" % ("gpt-4o", round_idx)]=openai_answer["raw"] # raw only for logging
            new_mag["%s_output_%d" % ("gpt-4o", round_idx)]={"reasoning": openai_answer["reasoning"], "answer": openai_answer["answer"]}
            token_cnt["gpt-4o"]["prompt_tokens"]=token_cnt["gpt-4o"]["prompt_tokens"]+openai_answer["prompt_tokens"]
            token_cnt["gpt-4o"]["completion_tokens"]=token_cnt["gpt-4o"]["completion_tokens"]+openai_answer["completion_tokens"]
            token_cnt["gpt-4o"]["cached_tokens"]=token_cnt["gpt-4o"]["cached_tokens"]+openai_answer["cached_tokens"]

        all_stop=True
        for new_mag in new_mags:
            if new_mag["stop"]:
                continue
            current_round_right={}
            for model_name in model_list:
                current_round_right[model_name]=right_or_wrong(new_mag["%s_output_%d" % (model_name, round_idx)]["answer"], new_mag["gold_answer"], args.dataset)
            new_mag["right_%d" % round_idx]=current_round_right
            if all(current_round_right.values()):
                new_mag["stop"]=True
            else:
                new_mag["stop"]=False
                all_stop=False
        if all_stop:
            break
            
        logger.info("round %d.5", round_idx)
        round_end(new_mags, round_idx)

        logger.info("--------------------------------------------------------")

        round_idx+=1

    logger.warning(json.dumps(token_cnt, ensure_ascii=False, indent=2))
    output_mode = 'w' if args.run_suffix else 'a'
    with open("./data/MAG_new_mistral/%s_1000%s.jsonl" % (args.dataset, args.run_suffix), mode=output_mode, encoding="utf-8") as fout:
        for new_mag in new_mags:
            fout.write(json.dumps(new_mag, ensure_ascii=False)+'\n')

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_questions", type=int, default=10, help="Limit number of questions to process (for robustness tests)")
    parser.add_argument("--run_suffix", type=str, default="", help="Suffix for output filename, e.g. '_5q'. Creates a separate file rather than appends.")
    args=parser.parse_args()

    main(args)
