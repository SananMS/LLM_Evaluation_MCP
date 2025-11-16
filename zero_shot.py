import json
from openai import OpenAI

client = OpenAI(api_key="INSERT_YOUR_API_KEY")

INPUT_FILE = "data/extracted_50.json"
OUT_MINI = "results/results_4.1_mini.json"
OUT_NANO = "results/results_4.1_nano.json"
DEBUG_LOG = "results/zero_shot_debug_log.txt"
MODEL_MINI = "gpt-4.1-mini"
MODEL_NANO = "gpt-4.1-nano"

def ask_model(model_name, question, options, q_index, correct):
    opts_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

    prompt = f"""
You are answering a single multiple-choice question.
STRICT RULES (follow EXACTLY):
- Output ONLY a single letter (A, B, C, D, or E).
- Do NOT output explanations.
- Do NOT output multiple letters.
- Do NOT output words.
- Do NOT output punctuation.
- ONLY output one character: A / B / C / D / E.

Question:
{question}

Options:
{opts_text}

Your answer (ONLY one letter A-E):
""".strip()

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
        dbg.write(f"QUESTION #: {q_index+1}\n")
        dbg.write("--------------- PROMPT ---------------\n")
        dbg.write(prompt + "\n\n")
        dbg.write("--------------- RAW OUTPUT ----------\n")
        dbg.write(raw_output + "\n\n")
        dbg.write("--------------- EXTRACTED ------------\n")
        dbg.write(f"Predicted: {predicted}\n")
        dbg.write(f"Correct:   {correct}\n")
        dbg.write("============================================================\n\n\n")

    return predicted

def run_eval(model_name, output_file):
    print(f"\n=== Running model: {model_name} ===\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    correct_count = 0

    for idx, q in enumerate(dataset):
        question = q["question"]
        options = q["options"]
        correct_answer = q["answer"]

        predicted = ask_model(
            model_name, question, options, idx, correct_answer
        )

        results.append({
            "question": question,
            "options": options,
            "correct": correct_answer,
            "predicted": predicted,
            "is_correct": predicted == correct_answer
        })

        if predicted == correct_answer:
            correct_count += 1

        print(f"{idx+1}. Predicted: {predicted} | Correct: {correct_answer}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nModel {model_name} scored {correct_count}/{len(dataset)} correct.\n")
    return correct_count

with open(DEBUG_LOG, "w", encoding="utf-8") as dbg:
    dbg.write("ZERO-SHOT DEBUG LOG\n\n")

score_mini = run_eval(MODEL_MINI, OUT_MINI)
score_nano = run_eval(MODEL_NANO, OUT_NANO)

print("================================")
print("     FINAL SUMMARY")
print("================================")
print(f"GPT-4.1-mini accuracy: {score_mini}/50")
print(f"GPT-4.1-nano accuracy: {score_nano}/50")
