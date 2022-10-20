import transformers
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenizing input text
text = "I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."
tokenized_text =tokenizer.encode(text)
print(tokenized_text)
# Creating a dummy input
dummy_input = torch.tensor(tokenized_text).unsqueeze(0)
print(dummy_input.shape)
MODELPATH="./models/v1"
model = AutoModelForSequenceClassification.from_pretrained(MODELPATH)#pad_token_id=tokenizer.eos_token_id)

# The model needs to be in evaluation mode
model.eval()

# Creating the trace
traced_model = torch.jit.trace(model, dummy_input,strict=False)
#TODO: make output not dictionary
torch.jit.save(traced_model, "dadv1.pt")

#model_scripted = torch.jit.script(model)
#model_scripted.save("./model_deployment/model_v1.pt")

class Daddy(torch.nn.Module):
    def __init__(self):
        super(Daddy, self).__init__()

        self.distilBert = AutoModelForSequenceClassification.from_pretrained(MODELPATH)
    def forward(self, x):
        #TOKENIZED INPUT
        x = self.distilBert(torch.tensor(x))
        x = model(x)['logits'] #TENSOR!
        return x

joke="I thought the dryer was shrinking my clothes. Turns out it was the refrigerator all along."
jokeTKN=tokenizer.encode(joke, return_tensors = "pt")

dad=Daddy()
print(dad(jokeTKN))

