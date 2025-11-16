import json

input_file = "data/train.jsonl"
output_file = "data/extracted_50.json"

extracted = []
limit = 50

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if len(extracted) >= limit:
            break
        
        line = line.strip()
        if not line:
            continue
        
        data = json.loads(line)

        extracted.append({
            "question": data.get("question", ""),
            "options": data.get("options", {}),
            "answer": data.get("answer_idx")  
        })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted, f, indent=4, ensure_ascii=False)

print("Saved 50 questions.")
