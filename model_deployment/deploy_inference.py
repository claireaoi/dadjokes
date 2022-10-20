from functools import cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage
import requests

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers import pipeline
import torch

test_dad_joke = "I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."
test_not_dad_joke = "What was Robin Hood's favourite variety of font? Sans-sheriff"

app = FastAPI()


def download_all_files_from_folder_in_bucket(bucket_name: str, folder_in_bucket: str):
    """
    Downloads all the contents of the bucket to the provided path.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_in_bucket)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name)


@cache
def load_model(bucket_name: str = "daddy-bucket", folder_in_bucket: str = "dad2"):
    # Download the contents of the bucket provided.
    if not Path(f"./{folder_in_bucket}").exists():
        download_all_files_from_folder_in_bucket(bucket_name, folder_in_bucket)

    model_path = f"./{folder_in_bucket}"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{model_path}/tokenizer.json")
    # tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer.json")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return model, tokenizer


def process(
    joke: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer
) -> torch.Tensor:
    """
    Tokenizes the joke, passes it through the model
    and returns the logits.
    """
    jokeTKN = tokenizer.encode(joke, return_tensors="pt")
    out = model(jokeTKN)["logits"]
    print(out.shape)

    return out


class PredictRequest(BaseModel):
    bucket_name: str  # gs://daddy-bucket
    model_version: str  # dad2
    input_joke: str


@app.post("/predict")
def predict(req: PredictRequest):
    classifier, tokenizer = load_model(req.bucket_name, req.model_version)

    classifier.eval()

    logits = process(req.input_joke, classifier, tokenizer)
    print(logits)
    probits = torch.softmax(logits, 1)
    print(probits)
    prediction = torch.argmax(logits)

    if prediction:
        answer = f"{req.input_joke} IS a dad joke :) "
    else:
        answer = f"{req.input_joke} IS NOT a dad joke :( "

    return {"AI opinion": answer, "confidence": probits[0, 1].item()}
