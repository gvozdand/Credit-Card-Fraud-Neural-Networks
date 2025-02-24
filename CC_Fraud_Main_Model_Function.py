from torch import nn
import torch.optim as optim
from CC_Fraud_NN_Functions import train_model, test_model
import matplotlib
matplotlib.use("Agg")
import torch

#Defining needed parameters and functions to chunk and observe data processing
batch_size = 32





def model_run(train_dataset, test_dataset, model_class, epochs, learn_rate):
    loss_list = []
    acc_list = []
    
    
    model = model_class()

    
    #defining optimizer for our models
    optimizer = optim.NAdam(model.parameters(), lr = learn_rate)
    
    #Loop trains and tests the model
    for iteration in range(epochs):
        print("-------- EPOCH: " + str((iteration+1)) + " --------")
        train_model(train_dataset, model, optimizer, loss_list)
        acc = test_model(test_dataset, model)
        acc_list.append(acc)
    torch.save(model.state_dict(), "temp_model_param.pth")
    print("Complete!")
    print(acc_list)
    return loss_list, acc_list
    












