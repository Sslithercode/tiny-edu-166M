from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./tiny-edu-166M-instruct-v3", trust_remote_code=True)
model.push_to_hub("SlitherCode/tiny-edu-166m-instruct-v3")