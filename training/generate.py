from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import pipeline
import torch

MODELPATH="./models/v1/"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODELPATH)#pad_token_id=tokenizer.eos_token_id)

joke="I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."

def test(joke):
    jokeTKN=tokenizer.encode(joke, return_tensors = "pt")
    out=model(jokeTKN)['logits']
    print(out.shape)
    return out

print(test(joke))


def evaluate(joke):
    classifier = pipeline("text-classification", model=MODELPATH, top_k=1)#return_all_scores=True)

    #NOTE: to return all score, use top_k=None
    out=classifier(joke)[0][0]['label']
    if out=='LABEL_1':
        return "not-dadjokes"
    else:
        return "dadjokes"

print(evaluate(joke))