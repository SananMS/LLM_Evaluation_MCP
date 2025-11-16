import json
from openai import OpenAI

client = OpenAI(api_key="INSERT_YOUR_API_KEY")

SHOT_COUNT = 10 # Number of few-shot examples to use

INPUT_FILE = "data/extracted_50.json"
OUT_MINI = f"results/results_mini_fewshot_{SHOT_COUNT}.json"
OUT_NANO = f"results/results_nano_fewshot_{SHOT_COUNT}.json"
DEBUG_LOG = f"results/debug_log_{SHOT_COUNT}.txt"
MODEL_MINI = "gpt-4.1-mini"
MODEL_NANO = "gpt-4.1-nano"


def build_fewshot_prompt(question, options, examples):
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


def ask_model(model_name, question, options, q_index, correct, examples):
    prompt = build_fewshot_prompt(question, options, examples)

    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    raw_output = response.output_text.strip()
    predicted = "?"
    for ch in raw_output.upper():
        if ch in ["A", "B", "C", "D", "E"]:
            predicted = ch
            break

    with open(DEBUG_LOG, "a", encoding="utf-8") as dbg:
        dbg.write("============================================================\n")
        dbg.write(f"MODEL: {model_name}\n")
        dbg.write(f"QUESTION #: {q_index}\n")
        dbg.write("--------------- PROMPT ---------------\n")
        dbg.write(prompt + "\n\n")
        dbg.write("--------------- RAW OUTPUT ----------\n")
        dbg.write(raw_output + "\n\n")
        dbg.write("--------------- PARSED OUTPUT -------\n")
        dbg.write(f"Predicted: {predicted}\n")
        dbg.write(f"Correct:   {correct}\n")
        dbg.write("============================================================\n\n\n")

    return predicted


def run_eval(model_name, output_file):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    examples = dataset[:SHOT_COUNT]
    eval_set = dataset[SHOT_COUNT:]

    results = []
    correct_total = 0

    for idx, q in enumerate(eval_set, start=1):
        predicted = ask_model(
            model_name,
            q["question"],
            q["options"],
            idx,
            q["answer"],
            examples
        )

        results.append({
            "question": q["question"],
            "options": q["options"],
            "correct": q["answer"],
            "predicted": predicted,
            "is_correct": predicted == q["answer"]
        })

        if predicted == q["answer"]:
            correct_total += 1

        print(f"{idx}. Predicted: {predicted} | Correct: {q['answer']}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nModel {model_name} scored {correct_total}/{len(eval_set)} correct.\n")
    return correct_total


with open(DEBUG_LOG, "w", encoding="utf-8") as dbg:
    dbg.write(f"DEBUG LOG FOR {SHOT_COUNT}-SHOT MCQ EVAL\n\n")

score_mini = run_eval(MODEL_MINI, OUT_MINI)
score_nano = run_eval(MODEL_NANO, OUT_NANO)

print("======================================")
print("            FINAL SUMMARY")
print("======================================")
print(f"{MODEL_MINI} accuracy: {score_mini}/{50-SHOT_COUNT}")
print(f"{MODEL_NANO} accuracy: {score_nano}/{50-SHOT_COUNT}")
