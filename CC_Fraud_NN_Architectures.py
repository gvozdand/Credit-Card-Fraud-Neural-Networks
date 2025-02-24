"""
Below are the various classical neural network architectures utilized for predicting credit card fraud detection.
"""

#Import necessary libraries
from torch import nn
import torch

def init_weights(mo):
   if isinstance(mo, nn.Linear):
       torch.nn.init.xavier_normal(mo.weight)

#Define neural network architeture through inheritance of Pytorch module class
#This neural network below is tailored for the dataset with more features
class More_Featured_Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        #There is one input layer, two hidden layers, and one output layer
        self.architecture = nn.Sequential(
            nn.Linear(30, 25), 
            nn.Tanh(),
            nn.Dropout(p = 0.5),
            nn.Linear(25, 25),
            nn.Tanh(),
            nn.Dropout(p = 0.5),
            nn.Linear(25, 10),
            nn.Tanh(),
            nn.Dropout(p = 0.5),
            nn.Linear(10, 2),
        )
        self.apply(init_weights)
    def forward(self, input_tensor):
        try:
            self.architecture.apply(init_weights)
            arch_passthrough = self.architecture(input_tensor)
            return arch_passthrough
        except RuntimeError:
            pass




#Creating instance of more featured model to be used later
mf_nn_model = More_Featured_Neural_Network()

#Check to ensure model intializes
print(mf_nn_model)


#Below is another neural network with the same number of layers but catered toward the dataset with less features
class Less_Featured_Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.lf_architecture = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Dropout(p = 0.5),
            nn.Linear(5, 10),
            nn.Tanh(),
            nn.Dropout(p = 0.5),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Dropout(p = 0.5),
            nn.Linear(10, 2)
        )
        self.apply(init_weights)
    def forward(self, input_tensor):
        try:
            self.lf_architecture.apply(init_weights)
            forward_prop = self.lf_architecture(input_tensor)
            return forward_prop
        except RuntimeError:
            pass

lf_nn_model = Less_Featured_Neural_Network()

print(lf_nn_model)




        
        
