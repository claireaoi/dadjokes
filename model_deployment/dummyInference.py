import argparse
import torch
import time
import numpy as np
import pickle
import pathlib
import yaml
import datetime

# class NN(torch.nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         self.linear_relu_stack = torch.nn.Sequential(
#             torch.nn.Linear(1, 10),
#             torch.nn.ReLU(),
#             torch.nn.Linear(10, 2),
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits

    
def predict(args):

    # joke = args['joke']   
    joke = torch.tensor(np.random.rand()).unsqueeze(dim=0)
    # model = NN()
    model = torch.jit.load(args['model'])
    # torch.save(model, "model_deployment/dummy_model.pt")
    # model_scripted = torch.jit.script(model)
    # model_scripted.save("model_deployment/dummy_model_jit.pt")
    model.eval()
    prediction = torch.argmax(model(joke))
    if prediction:
        print('\nYour data IS a dad joke\n')
    else:
        print('\nYour data IS NOT a dad joke\n')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--joke', type=str,  default='Hola', metavar='', help='')
    parser.add_argument('--model', type=str,  default='model_deployment/dummy_model_jit.pt', metavar='', help='')
    args = parser.parse_args()
    predict(vars(args))