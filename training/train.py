from datasets import load_dataset

imdb = load_dataset("imdb")

#NOTE ABOUT DATASET FORMAT imdb["test"][0]
#{
#    "label": 0,
#    "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
#}

from transformers import AutoTokenizer


#1---LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#Create a preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length (512 tokens)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#2---PREPROCESS CHECK

tokenized_imdb = imdb.map(preprocess_function, batched=True)

#Use DataCollatorWithPadding to create a batch of examples.
# It will also dynamically pad your text to the length of the longest element in its batch, so they are a uniform length. 
# While it is possible to pad your text in the tokenizer function by setting padding=True, dynamic padding is more efficient.

#3---DYNAMICALLY PAD TEXT

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#4---DYNAMICALLY PAD TEXT
#Load DistilBERT with AutoModelForSequenceClassification along with the number of expected labels:
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#5---CREATE MODEL
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

#6---DEFINE HYPERPARAMETERS FOR TRAINING
# Define your training hyperparameters in TrainingArguments.

training_args = TrainingArguments(
    output_dir="./models/v1", #The output directory
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)


#7---CREATE TRAINER
# Pass the training arguments to Trainer along with the model, dataset, tokenizer, and data collator.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#TRAINER CLASS https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.Trainer




#8---TRAIN TO FINE TUNE
#Call train() to fine-tune your model.

trainer.train()

#8---EVALUATE
trainer.evaluate()

#9---SAVE MODEL
trainer.save_model()


#9---TEST
