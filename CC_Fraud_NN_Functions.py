"""
Below are the necessary functions for operating forward and backward propogation
"""

#Import necessary libraries
from torch import nn
import torch


#Defining learning parameters
loss_function = nn.CrossEntropyLoss()

#Defining function to train the model
def train_model(training_set, desired_model, optimizer, loss_list):
    
    
    
    #Set the model to train mode for gradient purposes
    desired_model.train()
    
    #Loop containing forward and backward propagation
    for given_batch, (inpt, outpt) in enumerate(training_set):
        
        #Forward prop
        prediction = desired_model(inpt)
        
        #Disregards data where small errors halt program
        if outpt.long() is None:
            continue
        
        #Calculates loss to backward propagate
        loss = loss_function(prediction, outpt.long())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
#Defining function to test the model
def test_model(testing_set, desired_model):
    #Setting model to evaluation mode
    desired_model.eval()
    
    #Proceeds with obtaining accuracy metric for evaluating model performance
    correct = 0    
    with torch.no_grad():
        for inpt, outpt in testing_set:
            try:
                prediction = desired_model(inpt)
                correct += (prediction.argmax(1) == outpt.long()).type(torch.float).sum().item()
            except TypeError:
                continue
    correct /= len(testing_set.dataset)
    accuracy = 100*correct
    print("Accuracy: " + str(accuracy))
    return accuracy
    
    



