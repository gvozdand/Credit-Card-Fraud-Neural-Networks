"""
The script below creates a dashboard to practice varying specific parameters
"""

#Importing libraries
import panel as pn
import numpy as np
from CC_Fraud_Cleaner import full_train_loader, less_train_loader, full_test_loader, less_test_loader
from CC_Fraud_NN_Architectures import More_Featured_Neural_Network, Less_Featured_Neural_Network
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from CC_Fraud_Main_Model_Function import model_run
import seaborn as sns
import torch

#Creating dashboard class
class Dashboard:
    def __init__(self):
        
        #Dasboard initialization
        pn.extension()
        
    #Creates the necessary pages, texts, and other functions
    def page_creation(self):
        
        #Table of Contents
        table_of_contents_text = """
        1.) Motivation for Project
        --------------------------
        2.) Try it Yourself!
        --------------------------
        3.) Enriched Model
        --------------------------
        5.) Monte Carlo and GPU Support
        --------------------------
        """
        self.table_of_contents_page = pn.Column(table_of_contents_text)
        
        #Motivation Page
        
        motivation_text = """
        134 million Americans have fallen victim to credit card fraud, with 62 million of those individuals falling victim in the last year. Many banks employ various strategies to detect credit card fraud as soon as transactions occur. Models are difficult to concoct for such a prediction.
        
        This is where machine learning comes in! Machine learning, specifically neural networks, are often used for deeply nuanced modeling with incredible precision when a sufficient amount of data is available. This project is meant to display a series of neural network examples that banks may use to combat credit card fraud as transactions occur.
        
        The dataset being used is the European Credit Card Fraud dataset. This dataset contained 30 features and 1 output variable (fraud/no fraud). Nearly all of the features are converted in an indiscernable manner due to proprietary issues. As a result, it is impossible to interpret the features themselves and the values contained within. Despite this, these values can still be used for the neural network training process. However, whilst trying yourself, the dataset has been filtered to only the non-proprietary features: time since the first transaction in seconds and the amount of transaction.      
        
        Note: This project uses the Pytorch library for all of the neural network architecture and functions. The Pytorch documentation was used as a reference/guidance for construction throughout!
        """
        self.motivation_page = pn.Column(motivation_text, width = 600)
        
        
        #DIY Page
        do_it_yourself_text = """
        Below are various parameters that you can change to impact the training and prediction ability of the model. Here is a brief description of what each parameter means:
            
            -Epoch: The number of times a model is trained and/or tested (Suggested Value: 5-100)
            -Learning Rate: Determines how quickly a model will learn. Although it may seem that a larger number would be better, faster learning rates compel models to overfit. This prevents the model from being able to generalize to data it hasn't yet seen. This removes all utility of the model (Suggest Value: 0.000001 - 0.001)
            
        The main figures to observe are the accuracy and loss. They should be inversely related. Keep in mind, these two metrics are very primitive in terms of gauging the robustness of a model. Often times, they fail to indicate whether a model has been overfitted. Other metrics, such as an F1 score or Area Under Curve (AUC).
        
        Enjoy and try to get as high of an accuracy and low of a loss as possible! After the model finishes training, also offer it data to see what it will predict!
        
        """
        self.epoch_input = pn.widgets.IntSlider(name = "Number of Epochs", start = 0, end = 100, step = 1, value = 10)
        self.lr_input = pn.widgets.FloatInput(name = "Learning Rate", start = 0.0000001, end = 0.01, step = 0.001, value = 0.001)
        self.diy_button = pn.widgets.Button(name = "Run", button_type = "primary")
        self.graph_pane = pn.pane.Matplotlib(plt.figure(), tight = True)
        self.mesge = pn.pane.Str("", width = 400)
        self.pred_button = pn.widgets.Button(name = "Run", button = "primary")
        self.time_input = pn.widgets.IntInput(name = "Time from first Transaction", start = 0, end = 2000, step = 5, value = 5)
        self.amount_input = pn.widgets.FloatInput(name = "Transaction Amount", start = 0, end = 100000000, step = 100, value = 100)
        self.prediction = pn.pane.Str("", width = 400)
        
        def diy_click(action):
            self.mesge.object = "Loading! Training could take a few minutes! Scroll down for prediction when training is complete!"
            epochs = self.epoch_input.value
            learn_rt = self.lr_input.value
            loss_list, acc_list = model_run(less_train_loader, less_test_loader, Less_Featured_Neural_Network, epochs, learn_rt)
            self.mesge.object = ""
            plt.figure(figsize = (10, 5))
            sns.set_style("whitegrid")
            plt.subplot(1, 2, 1)
            sns.lineplot(x = np.arange(1, len(loss_list) + 1), y = loss_list, color = "blue", label = "Loss")
            sns.regplot(x = np.arange(1, len(loss_list) + 1), y = loss_list, scatter = False, color = "pink", line_kws={"linewidth": 3, "linestyle": "--"}, label = "Line of Best Fit")
            plt.title("Loss Over Iterations")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            
            plt.subplot(1, 2, 2)
            sns.lineplot(x = np.arange(1, len(acc_list) + 1), y = acc_list, color = "red", label = "Accuracy")
            plt.title("Accuracy Over Iterations")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            
            self.graph_pane.object = plt.gcf()
        self.diy_button.on_click(diy_click)
        
        def predict_click(action):
            temp_list_pre_tensor = []
            time = self.time_input.value
            amount = self.amount_input.value
            temp_list_pre_tensor.append(time)
            temp_list_pre_tensor.append(amount)
            temp_input_tensor = torch.tensor(temp_list_pre_tensor, dtype = torch.float32).unsqueeze(0)
            temp_model = Less_Featured_Neural_Network()
            temp_model.load_state_dict(torch.load("temp_model_param.pth"))
            temp_model.eval()
            pred = 0
            with torch.no_grad():
                prediction = temp_model(temp_input_tensor)
                prediction_class = torch.argmax(prediction, dim = 1)
            if prediction_class == 0:
                pred = "Not Fraud"
            else:
                pred = "Fraud"
            self.prediction.object = pred
        self.pred_button.on_click(predict_click)
            
            
        
        self.do_it_yourself = pn.Column(do_it_yourself_text, self.epoch_input, self.lr_input, self.mesge, self.diy_button, self.graph_pane, self.prediction, self.time_input, self.amount_input, self.pred_button, width = 600)

        enriched_model_text = """
        As mentioned earlier, the DIY model only worked with non-proprietary data. However, we can expand the dataset to utilize the proprietary information in efforts of creating an even more accurate and nuanced model. In doing so, we will add 28 features. Click train to observe the computations using the preset parameters!
        
        """
        
        self.enr_button = pn.widgets.Button(name = "Run", button_type = "primary")
        self.graph_pane_two = pn.pane.Matplotlib(plt.figure(), tight = True)
        self.mesge_two = pn.pane.Str("", width = 400)
        def enr_click(action):
            self.mesge_two.object = "Loading! Training could take a few minutes!"
            loss_list, acc_list = model_run(full_train_loader, full_test_loader, More_Featured_Neural_Network, 20, 0.00001)
            self.mesge_two.object = ""
            plt.figure(figsize = (10, 5))
            sns.set_style("whitegrid")
            plt.subplot(1, 2, 1)
            sns.lineplot(x = np.arange(1, len(loss_list) + 1), y = loss_list, color = "blue", label = "Loss")
            sns.regplot(x = np.arange(1, len(loss_list) + 1), y = loss_list, scatter = False, color = "pink", line_kws={"linewidth": 3, "linestyle": "--"}, label = "Line of Best Fit")
            plt.title("Loss Over Iterations")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            
            plt.subplot(1, 2, 2)
            sns.lineplot(x = np.arange(1, len(acc_list) + 1), y = acc_list, color = "red", label = "Accuracy")
            plt.title("Accuracy Over Iterations")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            
            self.graph_pane_two.object = plt.gcf()
        self.enr_button.on_click(enr_click)
        
        self.enriched_model = pn.Column(enriched_model_text, self.enr_button, self.mesge_two, self.graph_pane_two, width = 600)
        mc_gpu_text = """
        We have explored a small and medium scale version of these fraud prediction models. However, there are still ways to increase nuance detection and prediction power. We can do so by mixing more data, more neural network complexity, and more compute power:
            
            1. Although the European CC fraud dataset is capped at just south of 300,000 entries, we can simulate as much data as we would like to continue training these models. Specifically, a Monte Carlo simulation can be employed to generate this data.
            --------------
            2. We are also able to increase the complexity of the network architecture by adding more neurons and hidden layers. 
            --------------
            3. Lastly, using a GPU will allow for a considerable scale in compute power. GPUs can often outperform CPUs by 10-20 times. 
            --------------
            At this time, there is not a MC simulation. However, this feature will be available upon receiving access to a GPU in the near future!
            
        """
        self.monte_carlo_and_gpu = pn.Column(mc_gpu_text, width = 600)
        self.gpu_support = pn.Column("We can accelerate")
        
        self.tabs = pn.Tabs(
            ("Table of Contents", self.table_of_contents_page),
            ("Motivation", self.motivation_page),
            ("Try It Yourself!", self.do_it_yourself),
            ("Enriched Model", self.enriched_model),
            ("Monte Carlo Testing and GPU Acceleration", self.monte_carlo_and_gpu),
        )
    def display_dashboard(self):
        self.page_creation()
        self.do_it_yourself.show()
        self.tabs.show()
    
dashboard = Dashboard()
dashboard.display_dashboard()
