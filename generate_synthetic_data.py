erate mini dataset · PY
Copy

"""
Mini SFT dataset generator — 100 examples (25 per subject)
Uses distilabel + OpenRouter (Gemma 3 27B)
Pushes to HuggingFace Hub for inspection before scaling to 20k
 
Usage:
    pip install "distilabel[openai]" huggingface_hub
    export OPENROUTER_API_KEY="sk-or-..."
    export HF_TOKEN="hf_..."
    python generate_mini_dataset.py
"""
 
import os
import random
from datasets import Dataset
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import TextGeneration
 
# ── Config ────────────────────────────────────────────────────────────────────
 
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
HF_TOKEN           = os.environ["HF_TOKEN"]
HF_REPO_ID         = "your-username/sft-qa-mini"   # ← change to your HF username
 
MODEL              = "google/gemma-3-27b-it:free"   # free tier for test run
EXAMPLES_PER_SUBJ  = 25                             # 25 × 4 = 100 total
BATCH_SIZE         = 5
 
# ── Topic seeds (forces distribution across each subject) ─────────────────────
 
TOPICS = {
    "math": [
        "addition of two-digit numbers", "subtraction with borrowing",
        "multiplication tables", "long division", "fractions",
        "percentages", "basic algebra", "area of a rectangle",
        "perimeter", "prime numbers", "order of operations",
        "decimals", "ratios", "square roots", "negative numbers",
        "mean median mode", "basic probability", "exponents",
        "converting fractions to decimals", "word problems with money",
        "elapsed time", "Roman numerals", "factors and multiples",
        "basic geometry angles", "simple equations",
    ],
    "science": [
        "photosynthesis", "the water cycle", "Newton's laws of motion",
        "states of matter", "the solar system", "DNA and genetics",
        "food chains", "the periodic table", "volcanoes",
        "plate tectonics", "cell structure", "gravity",
        "ecosystems", "the human digestive system", "atoms and molecules",
        "weather and climate", "magnetism", "electrical circuits",
        "evolution", "the layers of the earth", "osmosis",
        "chemical reactions", "biodiversity", "the immune system",
        "speed velocity and acceleration",
    ],
    "history": [
        "the American Revolution", "World War I causes",
        "World War II key events", "the French Revolution",
        "the Roman Empire", "the Renaissance",
        "the Industrial Revolution", "the Cold War",
        "ancient Egypt", "the Civil Rights Movement",
        "the fall of the Berlin Wall", "the signing of the Magna Carta",
        "the discovery of America", "the Ottoman Empire",
        "the Great Depression", "the Boston Tea Party",
        "the Declaration of Independence", "ancient Greece",
        "the Space Race", "the Cuban Missile Crisis",
        "the Black Death", "the Silk Road",
        "the Vietnam War", "the Reformation", "ancient China",
    ],
    "english": [
        "the definition of a noun", "what is a verb",
        "the difference between an adjective and an adverb",
        "what is a metaphor", "what is a simile",
        "the definition of alliteration", "what is a synonym",
        "what is an antonym", "the definition of irony",
        "what is foreshadowing", "the difference between its and it's",
        "what is a pronoun", "the definition of a conjunction",
        "what is an idiom", "the definition of onomatopoeia",
        "what is a prefix", "what is a suffix",
        "the definition of a clause", "what is hyperbole",
        "the difference between active and passive voice",
        "what is a thesis statement", "the definition of an analogy",
        "what is personification", "the definition of a preposition",
        "what is a compound sentence",
    ],
}
 
# ── Build seed rows ────────────────────────────────────────────────────────────
 
def build_seed_rows(topics: dict, n_per_subject: int) -> list[dict]:
    rows = []
    for subject, topic_list in topics.items():
        sampled = random.sample(topic_list, min(n_per_subject, len(topic_list)))
        for topic in sampled:
            rows.append({
                "subject": subject,
                "topic": topic,
                "instruction": (
                    f"Generate one {subject} question and a concise, natural answer "
                    f"about: {topic}. "
                    f"The answer should be one sentence that directly answers the question. "
                    f"No explanations, no bullet points, just a question and its answer.\n\n"
                    f"Format your response exactly like this:\n"
                    f"Question: <your question>\n"
                    f"Answer: <your answer>"
                ),
            })
    random.shuffle(rows)
    return rows
 
# ── Parse question/answer from generation ─────────────────────────────────────
 
def parse_qa(generation: str) -> tuple[str, str]:
    """Extract Question and Answer from the model output."""
    lines = generation.strip().splitlines()
    question, answer = "", ""
    for line in lines:
        if line.lower().startswith("question:"):
            question = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("answer:"):
            answer = line.split(":", 1)[-1].strip()
    return question, answer
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    seed_rows = build_seed_rows(TOPICS, EXAMPLES_PER_SUBJ)
    print(f"Generating {len(seed_rows)} examples across 4 subjects...")
 
    llm = OpenAILLM(
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        generation_kwargs={
            "temperature": 0.9,   # some variety in phrasing
            "max_new_tokens": 150, # question + answer fits easily
        },
    )
 
    with Pipeline(name="sft-qa-mini") as pipeline:
        load = LoadDataFromDicts(data=seed_rows)
        generate = TextGeneration(
            llm=llm,
            input_batch_size=BATCH_SIZE,
            output_mappings={"generation": "raw_generation"},
        )
        keep = KeepColumns(columns=["subject", "topic", "instruction", "raw_generation"])
        load >> generate >> keep
 
    distiset = pipeline.run(use_cache=False)
 
    # ── Post-process: parse Q/A and build final dataset ───────────────────────
    raw_data = distiset["default"]["train"]
 
    final_rows = []
    skipped = 0
    for row in raw_data:
        gen = row.get("raw_generation") or ""
        question, answer = parse_qa(gen)
        if not question or not answer:
            skipped += 1
            continue
        final_rows.append({
            "subject":    row["subject"],
            "topic":      row["topic"],
            "question":   question,
            "answer":     answer,
            # SFT-ready messages format for TRL SFTTrainer
            "messages": [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
        })
 
    print(f"\nParsed: {len(final_rows)} valid  |  Skipped: {skipped}")
    print(f"\nSample:")
    for row in final_rows[:3]:
        print(f"  [{row['subject']}] Q: {row['question']}")
        print(f"           A: {row['answer']}\n")
 
    # ── Push to Hub ───────────────────────────────────────────────────────────
    ds = Dataset.from_list(final_rows)
    ds.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=True)
    print(f"\nPushed to: https://huggingface.co/datasets/{HF_REPO_ID}")
    print("Inspect the dataset on the Hub, then run generate_full_dataset.py to scale to 20k.")
 
if __name__ == "__main__":
    main()
 