from functools import cache
from fastapi import FastAPI
from pydantic import BaseModel
# from google.cloud import storage
import torch
import requests
# import re

app = FastAPI()

@cache
def load_model(model_path):
    # bucket, path = re.match(r"gs://([^/]+)/(.+)", model_path).groups()
    # clf_bytes = storage.Client().bucket(bucket).blob(path).download_as_bytes()
    _model = requests.get(model_path)
    open("model.pt", "wb").write(_model.content)
    model = torch.jit.load("model.pt")
    return model


class PredictRequest(BaseModel):
    model: str  # gs://path/to/model.pkl
    input_joke: str


@app.post("/predict")
def predict(req: PredictRequest):
    classifier = load_model(req.model) # so we can indicate the model url via REST
    # classifier = load_model('https://storage.googleapis.com/modelbucket1241241/dummy_model_jit.pt')
    classifier.eval()
    prediction = torch.argmax(classifier(req.input_joke))
    if prediction:
        answer = f'{req.input_joke} IS a dad joke :) '
    else:
        answer = f'{req.input_joke} IS NOT a dad joke :( '
    return {"AI opinion": answer}