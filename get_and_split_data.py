import json
from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro")
dataset=ds["test"]

computer_science_data=dataset.filter(lambda example: example["category"] == "computer science")
print(len(computer_science_data), "original test data")

computer_science_data_splited=computer_science_data.train_test_split(test_size=(1/2))
print(computer_science_data_splited)

with open("./data/MAG/MMLUProComp_1000.json", mode='w', encoding="utf-8") as fout:
    mags=[]
    for example in computer_science_data_splited["train"]:
        question=example["question"]
        qid=example["question_id"]
        options=example["options"]
        gold=example["answer"]
        q_o="%s" % question
        prefix=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for pr, o in zip(prefix, options):
            q_o+=(" %s) %s" % (pr, o))
        mag={"question": q_o, "gold_answer": gold, "MMLU_Pro_id": qid}
        mags.append(mag)
    json.dump(mags, fout, ensure_ascii=False, indent=2)
    print(len(mags))

with open("./data/test_data/MMLUProComp_test.json", mode='w', encoding="utf-8") as fout:
    mags=[]
    for example in computer_science_data_splited["test"]:
        question=example["question"]
        qid=example["question_id"]
        options=example["options"]
        gold=example["answer"]
        q_o="%s" % question
        prefix=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for pr, o in zip(prefix, options):
            q_o+=(" %s) %s" % (pr, o))
        mag={"question": q_o, "answer": gold, "MMLU_Pro_id": qid}
        mags.append(mag)
    json.dump(mags, fout, ensure_ascii=False, indent=2)
    print(len(mags))
