import transformers
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer

MODELPATH="./models/v1"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODELPATH)#pad_token_id=tokenizer.eos_token_id)

model_scripted = torch.jit.script(model)
model_scripted.save("model_deployment/model_v1.pt")

# class Daddy(torch.nn.Module):

#     def __init__(self):
#         super(Daddy, self).__init__()

#         self.linear1 = torch.nn.Linear(100, 200)
#         self.activation = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(200, 10)
#         self.softmax = torch.nn.Softmax()

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.linear2(x)
#         x = self.softmax(x)
#         return x


