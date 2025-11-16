import json
import random
from openai import OpenAI

client = OpenAI(api_key="INSERT_YOUR_API_KEY")

SHOT_COUNT = 5  # Few-shot examples
SHUFFLE_RUNS = 5  # Number of shuffle experiments

INPUT_FILE = "data/extracted_50.json"
OUTPUT_RESULTS = "results/results_shuffle_fewshot.json"
DEBUG_LOG = "results/shuffle_fewshot_debug_log.txt"
MODEL_MINI = "gpt-4.1-mini"


def build_fewshot_prompt(question, options, examples):
    """Build the few-shot prompt using the same structure as your original script."""

    example_blocks = []
    for ex in examples:
        ex_opts = "\n".join([f"{k}. {v}" for k, v in ex["options"].items()])
        example_blocks.append(
            f"Example Question:\n{ex['question']}\n\n"
            f"Options:\n{ex_opts}\n\n"
            f"Correct Answer: {ex['answer']}\n"
            "---------------------------------------\n"
        )

    target_opts = "\n".join([f"{k}. {v}" for k, v in options.items()])

    rules = (
        "STRICT RULES (follow EXACTLY):\n"
        "- Output ONLY a single letter (A, B, C, D, or E).\n"
        "- Do NOT output explanations.\n"
        "- Do NOT output multiple letters.\n"
        "- Do NOT output words.\n"
        "- Do NOT output punctuation.\n"
        "- ONLY output one character: A / B / C / D / E.\n"
    )

    prompt = (
        "Below are example questions with correct answers. Use them as demonstrations, then answer the final question.\n\n"
        + rules
        + "\n"
        + "\n".join(example_blocks)
        + "\nFINAL QUESTION:\n"
        + question
        + "\n\nOptions:\n"
        + target_opts
        + "\n\nFINAL ANSWER (A/B/C/D/E ONLY):"
    )

    return prompt


def ask_model(prompt):
    """Same style as your second file: returns raw + parsed prediction."""
    response = client.responses.create(
        model=MODEL_MINI,
        input=prompt
    )

    raw = response.output_text.strip()
    predicted = "?"

    for ch in raw.upper():
        if ch in ["A", "B", "C", "D", "E"]:
            predicted = ch
            break

    return raw, predicted


def run_single_shuffle(dataset, run_number):
    """Run one shuffle experiment."""
    base_examples = dataset[:SHOT_COUNT]
    eval_set = dataset[SHOT_COUNT:]

    shuffled_examples = base_examples.copy()
    random.shuffle(shuffled_examples)

    debug_entries = []
    run_results = []
    correct_count = 0

    for idx, item in enumerate(eval_set, start=1):
        question = item["question"]
        options = item["options"]
        correct = item["answer"]

        prompt = build_fewshot_prompt(question, options, shuffled_examples)

        raw, predicted = ask_model(prompt)
        is_correct = (predicted == correct)
        if is_correct:
            correct_count += 1

        run_results.append({
            "question": question,
            "correct": correct,
            "predicted": predicted,
            "is_correct": is_correct
        })

        debug_entries.append(
            "============================================================\n"
            f"SHUFFLE RUN: {run_number}\n"
            f"QUESTION #: {idx}\n"
            "--------------- PROMPT ---------------\n"
            f"{prompt}\n\n"
            "--------------- RAW OUTPUT -----------\n"
            f"{raw}\n\n"
            "--------------- PARSED OUTPUT --------\n"
            f"Predicted: {predicted}\n"
            f"Correct:   {correct}\n"
            "============================================================\n\n\n"
        )

        print(f"Run {run_number} | Q{idx}: Predicted={predicted} | Correct={correct}")

    return correct_count, run_results, debug_entries, shuffled_examples

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

all_runs_output = []

with open(DEBUG_LOG, "w", encoding="utf-8") as dbg:
    dbg.write("DEBUG LOG FOR SHUFFLED FEW-SHOT EXPERIMENT\n\n")

for run in range(1, SHUFFLE_RUNS + 1):
    print(f"\n=============== SHUFFLE RUN {run}/{SHUFFLE_RUNS} ===============")

    accuracy, run_results, debug_entries, example_order = run_single_shuffle(dataset, run)

    all_runs_output.append({
        "run": run,
        "accuracy": accuracy,
        "example_order": [ex["question"] for ex in example_order],
        "results": run_results
    })

    with open(DEBUG_LOG, "a", encoding="utf-8") as dbg:
        for entry in debug_entries:
            dbg.write(entry)

    print(f"Accuracy Run {run}: {accuracy}/{50 - SHOT_COUNT}")

with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
    json.dump(all_runs_output, f, indent=4, ensure_ascii=False)
