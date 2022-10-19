
#NOTE: need datasets, transformers, torch

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
#TODO:
#TODO: DATASET FORMAT
#TODO: TRAINING ARGUMENT
#TODO: IN CLOUD


#NOTE ABOUT DATASET FORMAT imdb["test"][0]
#{
#    "label": 0,
#    "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
#}
#dataset["train"][0]
#{'idx': 0,
# 'label': 1,
# 'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

MODELPATH="./models/v1"

from transformers import AutoTokenizer
import torch
torch.cuda.is_available() #CHECK

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################
######## PRELIMINARIES
################################################
print("#####PRELIMINARIES######")
#1---LOAD DATASET
from datasets import load_dataset

jokedata = load_dataset("imdb") #OLD ONE

# filePATH="/Users/clgl/Github/dadjokes/data_processing/output_jokes_joint.csv"
# dataset = load_dataset("csv", data_files=[filePATH], split="train")
# dataset = dataset.rename_column('joke', 'text')
# #dataset = dataset.rename_column('Unnamed: 0', 'idx')
# dataset = dataset.remove_columns("joke_length_in_words")
# dataset = dataset.remove_columns("Unnamed: 0")
# print(dataset.features)
# #NOTE: label should be  'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
# #SPLIT DATA
# jokedata=dataset.train_test_split(test_size=0.1, shuffle=True)
# print(jokedata["train"].features)

#1---LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#Create a preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length (512 tokens)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#2---PREPROCESS CHECK LENGTH
tokenized_jokes = jokedata.map(preprocess_function, batched=True)



#Use DataCollatorWithPadding to create a batch of examples.
# It will also dynamically pad your text to the length of the longest element in its batch, so they are a uniform length. 
# While it is possible to pad your text in the tokenizer function by setting padding=True, dynamic padding is more efficient.

#3---DYNAMICALLY PAD TEXT

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#5---CREATE MODEL
#Load DistilBERT with AutoModelForSequenceClassification along with the number of expected labels:
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

################################################
######## TRAINING
################################################
print("#####TRAINING######")

#6---DEFINE HYPERPARAMETERS FOR TRAINING
# Define your training hyperparameters in TrainingArguments.
#Trainer does not automatically evaluate model performance during training. You’ll need to pass Trainer a function to compute and report metrics. The 🤗 Evaluate library provides a simple accuracy function you can load with the evaluate.load (see this quicktour for more information) function:
import numpy as np
no_deprecation_warning=True

training_args = TrainingArguments(
    output_dir=MODELPATH, #The output directory
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    #eval_steps=1000, # Number of update steps between two evaluations.
    evaluation_strategy="epoch",
    save_steps=5000, # after # steps model is saved 
    weight_decay=0.01,
    warmup_steps=200# number of warmup steps for learning rate scheduler
)

#7---CREATE TRAINER
# Pass the training arguments to Trainer along with the model, dataset, tokenizer, and data collator.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_jokes["train"],
    eval_dataset=tokenized_jokes["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

#TRAINER CLASS https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.Trainer

#8---TRAIN, FINE TUNE
#Call train() to fine-tune your model.

trainer.train()

################################################
######## EVALUATE
################################################
print("#####EVALUATE######")
#TODO: COULD ADD EVALUATION https://huggingface.co/docs/transformers/training#finetune-with-trainer

#9---EVALUATE
trainer.evaluate()

#10---SAVE MODEL
trainer.save_model()



# ################################################
# ######## GENERATE
# ################################################
# print("#####TESTING######")

# #11---TEST


# joke="I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."

# #tokenize
# jokeTKN = tokenizer.encode(joke, return_tensors = "pt")
# #generate new line

# outputTKN = model.generate(jokeTKN, do_sample=True, temperature=0.1, max_length=100)
