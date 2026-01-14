from transformers import BartTokenizer, BartForConditionalGeneration
import torch

MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def rewrite_to_plain_language(text: str) -> str:
    prompt = (
        "Rewrite the following insurance rules into clear, simple language "
        "that a normal person can easily understand. "
        "Do not include section numbers, titles, or references.\n\n"
        + text
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
