from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import pipeline


MODELPATH="./models/v1/"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODELPATH)#pad_token_id=tokenizer.eos_token_id)

################################################
######## GENERATE
################################################
joke="I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."

print("#####TESTING joke {}######".format(joke))
print("*******TEST 1******")
jokeTKN = tokenizer.encode(joke, return_tensors = "pt")
output = model(jokeTKN)['logits'
amax=torch.amax(output, 1).item()
print(amax)


print("*******TEST 2******")
classifier = pipeline("text-classification",model=MODELPATH, return_all_scores=True)
print("output", classifier(joke))

