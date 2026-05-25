"""
Run locally to verify chat template + label masking before kicking off SFT.

    python test_masking.py

Needs: pip install tiktoken jinja2
No GPU, no HF token required — we replicate the tokenizer directly.
"""

import tiktoken
from jinja2 import Environment

enc = tiktoken.get_encoding("cl100k_base")

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ 'system\nYou are a helpful assistant<|endoftext|>\n' }}"
    "{% endif %}"
    "{{ message['role'] + '\n' + message['content'] + '<|endoftext|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant\n' }}{% endif %}"
)

def apply_chat_template(messages, add_generation_prompt=False):
    env = Environment()
    tmpl = env.from_string(CHAT_TEMPLATE)
    return tmpl.render(messages=messages, add_generation_prompt=add_generation_prompt)


ASSISTANT_HEADER = "assistant\n"
PAD_TOKEN_ID     = enc.eot_token  # <|endoftext|> == 100257


def tokenize(text, max_length=512):
    input_ids = enc.encode(text, allowed_special={"<|endoftext|>"})[:max_length]
    pad_len   = max_length - len(input_ids)
    input_ids = input_ids + [PAD_TOKEN_ID] * pad_len

    labels = [-100 if t == PAD_TOKEN_ID else t for t in input_ids]

    # mask prompt — everything up to and including "assistant\n"
    prompt_text = text.rsplit(ASSISTANT_HEADER, 1)[0] + ASSISTANT_HEADER
    prompt_len  = len(enc.encode(prompt_text, allowed_special={"<|endoftext|>"}))
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return input_ids, labels, prompt_len


tests = [
    {"instruction": "What is the capital of France?",
     "response":    "Paris.",
     "context":     ""},
    {"instruction": "What is 47 + 83?",
     "response":    "130",
     "context":     ""},
    {"instruction": "What is 123 * 456?",
     "response":    "56088",
     "context":     ""},
    {"instruction": "Who wrote Hamlet?",
     "response":    "William Shakespeare wrote Hamlet.",
     "context":     ""},
    # context field (dolly closed_qa style)
    {"instruction": "What city is described?",
     "response":    "Paris",
     "context":     "Paris is the capital of France."},
]

print(f"EOT/PAD token id: {PAD_TOKEN_ID}\n")
print("=" * 60)

all_ok = True
for t in tests:
    content = t["instruction"]
    if t["context"].strip():
        content = f"{content}\n\n{t['context'].strip()}"

    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": content},
        {"role": "assistant", "content": t["response"]},
    ]
    text = apply_chat_template(messages, add_generation_prompt=False)

    input_ids, labels, prompt_len = tokenize(text)

    supervised_ids  = [id for id in labels if id != -100]
    supervised_text = enc.decode(supervised_ids)
    pad_count       = input_ids.count(PAD_TOKEN_ID)

    response_present = t["response"] in supervised_text
    has_supervision  = len(supervised_ids) > 0
    ok = response_present and has_supervision

    print(f"{'✓' if ok else '✗'} {t['instruction']}")
    print(f"  full text:        {repr(text)}")
    print(f"  prompt masked:    {prompt_len} tokens")
    print(f"  pad tokens:       {pad_count}")
    print(f"  supervised ids:   {len(supervised_ids)} tokens")
    print(f"  supervised text:  {repr(supervised_text)}")
    if not ok:
        print(f"  !! response '{t['response']}' not found in supervised text")
    print()

    all_ok = all_ok and ok

print("=" * 60)
print("✓ All checks passed — masking is correct." if all_ok else "✗ FAILURES — fix before training.")