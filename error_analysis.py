"""
Error analysis script for D&R pipeline evaluation.
Loads test predictions and gold answers, categorizes errors, and prints examples.

Usage:
    python error_analysis.py --dataset MMLUProComp --results_dir ./tmp_test/<folder_name>
"""
import argparse
import json
from collections import Counter

from datasets import load_from_disk


def classify_error(response, pred_answer, gold_answer):
    """Classify the type of error based on the model's response."""
    if pred_answer == gold_answer:
        return "correct"

    # No answer extracted at all
    if pred_answer is None:
        if "Final answer:" not in response:
            return "format_error_no_final_answer"
        else:
            return "format_error_unparseable"

    # Model gave an answer but it's wrong
    # Check if the response is very short (likely didn't reason)
    reasoning_part = response.split("Final answer:")[0] if "Final answer:" in response else response
    reasoning_words = len(reasoning_part.split())

    if reasoning_words < 20:
        return "shallow_reasoning"

    # Check if gold answer letter appears anywhere in the reasoning (model considered it but picked wrong)
    if gold_answer in reasoning_part:
        return "wrong_selection"  # Considered the right answer but picked another
    else:
        return "wrong_reasoning"  # Never even considered the right answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the saved test results (e.g., ./tmp_test/MMLUProComp_qwen_...)")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of success/failure examples to print")
    args = parser.parse_args()

    # Load predictions
    preds = load_from_disk(args.results_dir)

    # Load gold answers
    with open(f"./data/test_data/{args.dataset}_test.json", "r") as f:
        test_samples = json.load(f)

    assert len(preds) == len(test_samples), f"Mismatch: {len(preds)} predictions vs {len(test_samples)} gold"

    # Classify each example
    results = []
    for i in range(len(preds)):
        pred_answer = preds[i]["answer"]
        gold_answer = test_samples[i]["answer"]
        response = preds[i]["response"]
        question = preds[i]["question"] if "question" in preds.column_names else test_samples[i]["question"]
        error_type = classify_error(response, pred_answer, gold_answer)

        results.append({
            "idx": i,
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "response": response,
            "error_type": error_type,
        })

    # --- Summary ---
    total = len(results)
    correct = sum(1 for r in results if r["error_type"] == "correct")
    print("=" * 80)
    print(f"ERROR ANALYSIS — {args.dataset}")
    print(f"Total: {total} | Correct: {correct} | Wrong: {total - correct} | Accuracy: {correct/total:.4f}")
    print("=" * 80)

    # --- Error categorization ---
    error_counts = Counter(r["error_type"] for r in results if r["error_type"] != "correct")
    print("\n### Error Categorization ###")
    print(f"{'Category':<35} {'Count':>5} {'% of Errors':>12}")
    print("-" * 55)
    total_errors = sum(error_counts.values())
    for cat, count in error_counts.most_common():
        pct = count / total_errors * 100 if total_errors > 0 else 0
        print(f"{cat:<35} {count:>5} {pct:>11.1f}%")
    print()

    ERROR_DESCRIPTIONS = {
        "format_error_no_final_answer": "Model did not output 'Final answer:' at all",
        "format_error_unparseable": "Model wrote 'Final answer:' but answer could not be parsed",
        "shallow_reasoning": "Model produced very little reasoning (<20 words) before answering",
        "wrong_selection": "Model discussed the correct answer in reasoning but selected a different one",
        "wrong_reasoning": "Model never considered the correct answer — flawed reasoning chain",
    }
    print("### Category Descriptions ###")
    for cat in error_counts:
        print(f"  - {cat}: {ERROR_DESCRIPTIONS.get(cat, 'Unknown')}")
    print()

    # --- Concrete examples ---
    successes = [r for r in results if r["error_type"] == "correct"]
    failures = [r for r in results if r["error_type"] != "correct"]

    n = args.num_examples

    print("=" * 80)
    print(f"### SUCCESS EXAMPLES (showing {min(n, len(successes))}) ###")
    print("=" * 80)
    for r in successes[:n]:
        print(f"\n--- Example #{r['idx']} | Gold: {r['gold']} | Pred: {r['pred']} ---")
        q_short = r["question"][:200] + "..." if len(r["question"]) > 200 else r["question"]
        print(f"Question: {q_short}")
        resp_short = r["response"][:400] + "..." if len(r["response"]) > 400 else r["response"]
        print(f"Response: {resp_short}")
        print()

    print("=" * 80)
    print(f"### FAILURE EXAMPLES (showing {min(n, len(failures))}, diverse categories) ###")
    print("=" * 80)

    # Try to pick failures from different categories
    shown_cats = set()
    diverse_failures = []
    for r in failures:
        if r["error_type"] not in shown_cats:
            diverse_failures.append(r)
            shown_cats.add(r["error_type"])
        if len(diverse_failures) >= n:
            break
    # Fill remaining slots if not enough categories
    for r in failures:
        if r not in diverse_failures:
            diverse_failures.append(r)
        if len(diverse_failures) >= n:
            break

    for r in diverse_failures[:n]:
        print(f"\n--- Example #{r['idx']} | Gold: {r['gold']} | Pred: {r['pred']} | Error: {r['error_type']} ---")
        q_short = r["question"][:200] + "..." if len(r["question"]) > 200 else r["question"]
        print(f"Question: {q_short}")
        resp_short = r["response"][:400] + "..." if len(r["response"]) > 400 else r["response"]
        print(f"Response: {resp_short}")
        print()


if __name__ == "__main__":
    main()
