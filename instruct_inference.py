import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./tiny-edu-166M-instruct-v3"  # update to your new checkpoint folder

tokenizer = AutoTokenizer.from_pretrained("SlitherCode/tiny-edu-166m", trust_remote_code=True)
tokenizer.pad_token = "<|endofprompt|>"
PAD_ID = tokenizer.convert_tokens_to_ids("<|endofprompt|>")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()

questions = [
    # arithmetic
    "2790 + 6698 =",
    "12 + 5 =",
    "1 + 2 =",
    # in-distribution QA
    "Who wrote Romeo and Juliet?",
    "What is the capital of France?",
    # out of distribution
    "What is the capital of Germany?",
    "Who invented the telephone?",
    "write a short poem",
]

def ask(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            repetition_penalty=1.0,
            temperature=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=PAD_ID,
        )

    raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
    response = raw.split("<|endoftext|>")[0].strip()
    return response

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print()