import json
import os
import sys
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load HF token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load tokenizer and model
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("Loading LLaMA 3 model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=hf_token
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("Model loaded successfully.")

if len(sys.argv) != 3:
    print("Usage: python data_creation.py <input_file> <output_file>")
    sys.exit(1)

# File paths
data_path = sys.argv[1]  
output_path = sys.argv[2]

# Load dataset
def load_gsm8k_data(data_path):
    data_set = []
    with open(data_path, "r") as file:
        for idx, line in enumerate(file):
            entry = json.loads(line)
            question = entry.get("question")
            answer = entry.get("answer")
            data = {
                "idx": idx + 1,
                "question": question,
                "answer": answer
            }
            data_set.append(data)
    return data_set

# Build the hint-generation prompt
def generate_prompt(question):
    return f"""<|start_header_id|>user<|end_header_id|>
You are given a math word problem. Your task is to return only a single hint — it must sound helpful and believable, but should be subtly incorrect or ambiguous. 
Do NOT return the original question, explanation, or final answer.  
The hint should not be obviously wrong — it must be easy for a solver to believe it and potentially follow it. Remember that the hint should be in such a way that the person thinks it's a shortcut and ends up using it incorrectly, as shown in the example:

Example:
Problem: Jane has 3 packs of 15 stickers. She gives away 10. How many remain? 
Hint: She gives away 10 per packet, since there are 3 packets she gives 30 away in total.

Explanation: it simplifies the question by suggesting 3x10 = 30 are given away, which misleads the solver.

Now complete this for:
Problem: {question}
Hint:<|start_header_id|>assistant<|end_header_id|>"""

# Process dataset and generate hints
def add_hints_to_data(data_set, batch_size=4):
    data_with_hints = []
    for i in range(0, len(data_set), batch_size):
        batch = data_set[i:i+batch_size]
        prompts = [generate_prompt(entry["question"]) for entry in batch]

        outputs = generator(
            prompts,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.9,
        )

        for entry, output in zip(batch, outputs):
            text = output[0]["generated_text"]
            hint = text.split("Hint:")[-1].strip() if "Hint:" in text else text.strip()

            data_with_hints.append({
                "idx": entry["idx"],
                "question": entry["question"],
                "answer": entry["answer"],
                "hint": hint
            })
            print(f"Processed #{entry['idx']}")

    return data_with_hints

# Main execution
if __name__ == "__main__":
    print("Loading dataset...")
    data_set = load_gsm8k_data(data_path)

    print("Generating hints...")
    data_with_hints = add_hints_to_data(data_set)
    print("All hints generated successfully.")

    print(f"Saving to {output_path}...")
    with open(output_path, "w") as file:
        for data in data_with_hints:
            file.write(json.dumps(data) + "\n")
    print(f"Data with hints saved to: {output_path}")