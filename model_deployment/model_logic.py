"""
Utility functions 
"""

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import pipeline
import torch

# TODO: update this tokenizer.
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODELPATH
# )  # pad_token_id=tokenizer.eos_token_id)


def process(joke: str, tokenizer, model) -> torch.Tensor:
    """
    Tokenizes the joke, passes it through the model
    and returns the logits.
    """
    jokeTKN = tokenizer.encode(joke, return_tensors="pt")
    out = model(jokeTKN)["logits"]
    print(out.shape)

    return out


if __name__ == "__main__":
    # TODO: update this tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Loading the model

    pass
