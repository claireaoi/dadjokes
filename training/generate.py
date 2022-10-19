from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

MODELPATH="distilbert-base-uncased"#"./models/v1/"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODELPATH)#pad_token_id=tokenizer.eos_token_id)


################################################
######## GENERATE
################################################

#11---TEST
joke="I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."
print("#####TESTING joke {}######".format(joke))

#NEED tokenize?
#jokeTKN = tokenizer.encode(joke, return_tensors = "pt")
#NEED PIPELINE ?
#classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

outputTKN = model(joke)#max_length?

#WHAT IS THE OUTPUT FORMAT ?
# Output:
# [[
# {'label': 'sadness', 'score': 0.0006792712374590337}, 
# {'label': 'joy', 'score': 0.9959300756454468}, 
# {'label': 'love', 'score': 0.0009452480007894337}, 
# {'label': 'anger', 'score': 0.0018055217806249857}, 
# {'label': 'fear', 'score': 0.00041110432357527316}, 
# {'label': 'surprise', 'score': 0.0002288572577526793}
# ]]

print(outputTKN)
