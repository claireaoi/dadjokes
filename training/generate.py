from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

MODELPATH="./models/v1/"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODELPATH)#pad_token_id=tokenizer.eos_token_id)



################################################
######## GENERATE
################################################

#11---TEST
joke="I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."
print("#####TESTING joke {}######".format(joke))


#tokenize
jokeTKN = tokenizer.encode(joke, return_tensors = "pt")
#generate new line

outputTKN = model.generate(jokeTKN, do_sample=True, temperature=0.1,)#max_length?
