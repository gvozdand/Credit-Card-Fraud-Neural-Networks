"""
This script below cleans and prepares the European Credit Card Fraud dataset for the subsequent neural networks.
"""

#Needed libraries for reading and manipulating data
import pandas as p
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def clean_data():
    show_checks = False
    
    #initial CSV to Dataframe conversion
    initial_df = p.read_csv(r"C:\Users\gvozd\OneDrive\Desktop\CC Fraud Dataset.csv")
    
    #Removal of proprietary features to create second smaller dataset
    features_to_be_dropped = [f"V{i}" for i in range(1, 29)]
    df_without_V_features = initial_df.drop(columns = features_to_be_dropped)
    
    if show_checks == True:
        
        #Check to ensure there is difference in dataframes
        print(initial_df)
        print(df_without_V_features)
        
        #Checks for NaN values => printed False which indicates clean data in that regard
        print(initial_df.isnull().values.any())
        print(df_without_V_features.isnull().values.any())
        
    #Removes any duplicate rows
    initial_df.drop_duplicates(inplace=True)
    df_without_V_features.drop_duplicates(inplace=True)
    
    if show_checks == True:
        
        #Check for number of columns and features
        print(len(initial_df.columns)) #returns 31 total columns/features
        print(len(df_without_V_features.columns)) #returns 3 total columns/features
        
    #Function ultimately returns cleaned data
    return initial_df, df_without_V_features

#Assigning variables to output of cleaning function
fully_featured_df, lesser_featured_df = clean_data()


def data_manipulation(cleaned_data, batch_size):
    
    #Removes dependent variable from df, leaving only features needed for input tensor
    features_df = cleaned_data.drop(columns=["Class"])
    
    #Removes all features except for dependent variable that will be turned into tensor for training
    output_df = cleaned_data["Class"]
    
    #Converts altered dataframes into input and output tensors
    input_tensor = torch.tensor(features_df.values, dtype = torch.float32)
    output_tensor = torch.tensor(output_df.values, dtype = torch.float32)

    #Splits the tensors into sets of training and testing tensors, each with input and output characteristics
    train_inpt, test_inpt, train_outpt, test_outpt = train_test_split(input_tensor, output_tensor, test_size = 0.2, random_state = 50)
    
    #Creates training and testing batches from obtained tensors above
    train_dataset = TensorDataset(train_inpt, train_outpt)
    train_ds_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataset = TensorDataset(test_inpt, test_outpt)
    test_ds_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    #Function returns two sets of batches derived from tensors
    return train_ds_loader, test_ds_loader

#Assigning output of manipulation function to separate variables for each dataset and each purpose (training/testing)
full_train_loader, full_test_loader = data_manipulation(fully_featured_df, 32)
less_train_loader, less_test_loader = data_manipulation(lesser_featured_df, 32)

