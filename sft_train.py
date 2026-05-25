import modal

app = modal.App("tiny-edu-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "tiktoken",
        "huggingface_hub",
        "wandb",
    )
)

volume = modal.Volume.from_name("tiny-edu-sft", create_if_missing=True)
VOLUME_PATH = "/vol"

ASSISTANT_HEADER = "assistant\n"
MAX_LENGTH = 1024
PAD_TOKEN = "<|endofprompt|>"


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 5,
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train():
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import load_dataset, concatenate_datasets
    import torch

    MODEL_ID = "SlitherCode/tiny-edu-166m"
    OUTPUT_DIR = f"{VOLUME_PATH}/checkpoints"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # use <|endofprompt|> as pad so EOT stays in labels as a learnable stop signal
    tokenizer.pad_token = PAD_TOKEN
    PAD_ID = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    model.config.pad_token_id = PAD_ID

    # ── Datasets ──────────────────────────────────────────────────────────────

    # Dolly QA
    dolly = load_dataset("Cleanlab/databricks-dolly-15k-cleaned", split="train")
    qa_categories = {"closed_qa", "open_qa", "information_extraction"}
    dolly = dolly.filter(lambda x: x["category"] in qa_categories)
    dolly = dolly.filter(lambda x:
        x.get("instruction") and x["instruction"].strip() and
        x.get("response") and x["response"].strip()
    )
    print(f"Dolly QA subset after cleaning: {len(dolly)} examples")

    # SimpleMath — 2500 per operation, balanced
    math_ds = load_dataset("ProCreations/SimpleMath", split="train")
    print(f"SimpleMath columns: {math_ds.column_names}")
    print(f"SimpleMath example: {math_ds[0]}")

    # sample 2500 per operator, balanced across +, -, x, /
    math_subsets = []
    for op in ["+", "-", "*", "/"]:
        subset = math_ds.filter(lambda x, o=op: o in x["problem"])
        subset = subset.shuffle(seed=42).select(range(min(2500, len(subset))))
        math_subsets.append(subset)
        print(f"  op={op}: {len(subset)} examples")
    math_balanced = concatenate_datasets(math_subsets)
    print(f"SimpleMath balanced total: {len(math_balanced)} examples")

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_dolly(example):
        instruction = example["instruction"]
        if example.get("context") and example["context"].strip():
            instruction = f"{instruction}\n\n{example['context'].strip()}"
        messages = [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": example["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    def format_math(example):
        # SimpleMath has "question" and "answer" columns
        messages = [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": example["problem"]},
            {"role": "assistant", "content": str(example["answer"])},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    dolly_fmt = dolly.map(format_dolly, remove_columns=dolly.column_names)
    math_fmt  = math_balanced.map(format_math, remove_columns=math_balanced.column_names)

    dataset = concatenate_datasets([dolly_fmt, math_fmt]).shuffle(seed=42)
    print(f"Total examples: {len(dataset)}")

    # ── Tokenize with completion-only loss ────────────────────────────────────

    def tokenize(example):
        full = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        input_ids = full["input_ids"]

        # mask only pad tokens — EOT stays in labels as stop signal
        labels = [-100 if t == PAD_ID else t for t in input_ids]

        # mask prompt up to and including "assistant\n"
        prompt_text = example["text"].rsplit(ASSISTANT_HEADER, 1)[0] + ASSISTANT_HEADER
        prompt_len  = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        full["labels"] = labels
        return full

    dataset = dataset.map(tokenize, remove_columns=["text"])
    dataset.set_format("torch")

    # ── Training ──────────────────────────────────────────────────────────────

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        report_to="wandb",
        run_name="tiny-edu-sft-v3",
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(f"{VOLUME_PATH}/final")
    tokenizer.save_pretrained(f"{VOLUME_PATH}/final")
    volume.commit()
    print("Done! Model saved to volume.")


@app.local_entrypoint()
def main():
    train.remote()