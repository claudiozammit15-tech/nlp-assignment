"""
Generate MMLUProHist test data for domain shift evaluation.
Downloads MMLU-Pro, filters to 'history', and creates test-only split.
We only need the test set (no debate/training on this domain).
"""
import json
from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro")
dataset = ds["test"]

hist_data = dataset.filter(lambda example: example["category"] == "history")
print(len(hist_data), "history questions in MMLU-Pro")

# We only need a test set for domain shift evaluation (no training)
# Take all of them, or cap at a reasonable size
with open("./data/test_data/MMLUProHist_test.json", mode='w', encoding="utf-8") as fout:
    mags = []
    for example in hist_data:
        question = example["question"]
        qid = example["question_id"]
        options = example["options"]
        gold = example["answer"]
        q_o = "%s" % question
        prefix = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for pr, o in zip(prefix, options):
            q_o += (" %s) %s" % (pr, o))
        mag = {"question": q_o, "answer": gold, "MMLU_Pro_id": qid}
        mags.append(mag)
    json.dump(mags, fout, ensure_ascii=False, indent=2)
    print(f"Wrote {len(mags)} history test questions to ./data/test_data/MMLUProHist_test.json")
