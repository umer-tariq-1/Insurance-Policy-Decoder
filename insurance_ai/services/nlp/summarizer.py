# insurance_ai/services/nlp/summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def summarize_chunk(text: str) -> str:
    prompt = (
        "Rewrite the following text as clear factual points for a policyholder. "
        "Do not mention page numbers, document titles, or instructions. "
        "Only state the actual rules, conditions, or facts.\n\n"
        + text
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
