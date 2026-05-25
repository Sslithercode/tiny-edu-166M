import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SlitherCode/tiny-edu-166m", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./model_v1", trust_remote_code=True)
model.eval()

def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== Prompt ===")
print(prompt)

# Proper generation
inputs = tokenizer(prompt, return_tensors="pt")
input_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode only new tokens, not the prompt
response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
print("=== Response ===")
print(response)