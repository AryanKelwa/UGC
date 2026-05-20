import json
import os
import random

SYSTEM_PROMPT = (
    "You are a real estate lead qualification assistant for Anywhere Real Estate (operating brands including Coldwell Banker, Century 21, ERA, Sotheby's International Realty, Better Homes and Gardens Real Estate, and Corcoran). Given a lead conversation, output a JSON with: intent (buy/rent/invest/browse), urgency (hot/warm/cold), budget_range, timeline_months, preferred_locality, property_type, brand_fit (which Anywhere brand best matches the lead), next_action (agent_call/schedule_showing/follow_up_email/nurture_sequence/drop), score (0–100), and reasoning."
)

data_dir = r"d:\Aryan Kelwa\Aryan\Internship\UGC\training-data-clean"
output_dir = r"d:\Aryan Kelwa\Aryan\Internship\UGC"

all_data = []

# Merge all 34 batches
for i in range(1, 35):
    filename = f"batch_{i:03d}.json"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
            all_data.extend(raw)
    else:
        print(f"Warning: {filepath} not found.")

print(f"Total merged examples: {len(all_data)}")

# Convert to required format
converted = []
for item in all_data:
    converted.append({
        "conversations": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": item["user"]},
            {"role": "assistant","content": item["assistant"]}
        ]
    })

# Shuffle the data
random.seed(42)
random.shuffle(converted)

# Split 70, 15, 15
total = len(converted)
train_split = int(0.70 * total)
eval_split = int(0.85 * total)

train_data = converted[:train_split]
eval_data = converted[train_split:eval_split]
test_data = converted[eval_split:]

print(f"Train size: {len(train_data)}")
print(f"Eval size: {len(eval_data)}")
print(f"Test size: {len(test_data)}")

# Save to JSONL
def save_jsonl(data, filename):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

save_jsonl(train_data, "train_data.jsonl")
save_jsonl(eval_data, "eval_data.jsonl")
save_jsonl(test_data, "test_data.jsonl")

print("Data preparation complete.")
