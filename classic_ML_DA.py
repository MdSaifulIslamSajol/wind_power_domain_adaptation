#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 00:44:54 2024

@author: saiful
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3 "
import argparse, os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from model.model import * 
from dataloader.dataloader import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pdb
import copy
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score


SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
target_data = "DE"
source_data = "UK"  # will be adapted from
#%%
def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

# Define the neural network model
class Wind_Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Wind_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 1, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.batchnorm1(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.batchnorm2(x4)
        x6 = self.relu(x5)
        x7 = self.flatten(x6)
        x8 = self.fc1(x7)
        x9 = self.relu(x8)
        x10 = self.fc2(x9)
        return x10
#%% FR
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_FR2.csv'
df_FR = pd.read_csv(file_path)
column_names = df_FR.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = df_FR['FR'].min()
max_value = df_FR['FR'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_FR = np.histogram(df_FR['FR'], bins=num_bins, range=(min_value, max_value))
print("Bin bin_edges_FR:", bin_edges_FR)

# Plot the histogram
plt.hist(df_FR['FR'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of FR column')
plt.xlabel('FR values')
plt.ylabel('Frequency')
plt.show()
#%%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data

# Split features and labels

#%%
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_UK2.csv'
df_UK = pd.read_csv(file_path)

file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv'
df_DE = pd.read_csv(file_path)

file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_FR2.csv'
df_FR = pd.read_csv(file_path)


# Assuming df is your DataFrame
selected_indices = [1, 3, 8, 9, 10, 13]
# Selecting the index column and last column
df_FR = df_FR.iloc[:, selected_indices + [-1]]
df_DE = df_DE.iloc[:, selected_indices + [-1]]
df_UK = df_UK.iloc[:, selected_indices + [-1]]


if target_data== 'FR':
    
    df = df_FR
    df.dropna(subset=['FR'], inplace=True)
    X = df.drop('FR', axis=1)  # Features
    y = df['FR']  # Labels
elif target_data== 'DE':
    
    df = df_DE
    df.dropna(subset=['DE'], inplace=True)
    X = df.drop('DE', axis=1)  # Features
    y = df['DE']  # Labels
    
elif target_data== 'UK':
    df = df_UK
    df.dropna(subset=['UK'], inplace=True)
    X = df.drop('UK', axis=1)  # Features
    y = df['UK']  # Labels
    


# Define bins for class labels
data_bin = [1.0000e-05, 1.6422e-01, 3.2843e-01, 4.9264e-01, 6.5685e-01, 8.2106e-01, 9.8527e-01]

# Define labels for the bins
labels = [0, 1, 2, 3, 4, 5]

# Replace the continuous labels with classes
y = pd.cut(y, bins=data_bin, labels=labels)

nan_count = y.isna().sum()
y = y.fillna(0)
#%%%%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Train the SVM model
# svm_model = SVC(kernel='linear')  # You can choose different kernels like 'rbf' or 'poly' as well
# svm_model.fit(X_train_scaled, y_train)
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Convert your pandas DataFrames to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # Assuming y_train is a pandas Series
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)  # Assuming y_test is a pandas Series

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model instance
model = Wind_Model(input_size=6, num_classes=6)  # Assuming input size and number of classes

# Load the saved model state dictionary
model_name = source_data+'_wind_model.pth'
# model.load_state_dict(torch.load('wind_model.pth'))
model.load_state_dict(torch.load(model_name))

model = model.to(device)
model = configure_model(model)

# model = configure_model(model)

# Ensure the model is in evaluation mode
model.eval()
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Calculate testing accuracy
correct_predictions = 0
total_samples = 0

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs.unsqueeze(2))
    _, predicted = torch.max(outputs, 1)
    correct_predictions += (predicted == labels).sum().item()
    total_samples += labels.size(0)

testing_accuracy = correct_predictions / total_samples
print("Testing Accuracy Before Domain Adaptation:", testing_accuracy)

#%% FINE TUNNING
# Fine-tune the model
num_epochs = 300  # You can adjust the number of epochs for fine-tuning
test_accuracy_values = []
epoch_list =[]
train_accuracy_values = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs.unsqueeze(2))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    train_accuracy_values.append(epoch_accuracy) 
    

    print(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy on new dataset: {epoch_accuracy}")
    # Calculate testing accuracy every 5 epochs
    if (epoch + 1) % 1 == 0:
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.unsqueeze(2))
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        testing_accuracy = correct_predictions / total_samples
        test_accuracy_values.append(testing_accuracy) 
        epoch_list.append(epoch + 1)

        print(f"Testing Accuracy after {epoch+1} epochs:", testing_accuracy)

#%%
# # Plotting test accuracy
# plt.plot(range(0, num_epochs + 1), test_accuracy_values, marker='o')
# plt.title('Testing Accuracy vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Testing Accuracy')
# plt.grid(True)
# plt.show()
#%%
# Calculate testing accuracy
correct_predictions = 0
total_samples = 0

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs.unsqueeze(2))
    _, predicted = torch.max(outputs, 1)
    correct_predictions += (predicted == labels).sum().item()
    total_samples += labels.size(0)

testing_accuracy = correct_predictions / total_samples
print("Testing Accuracy after Domain Adaptation:", testing_accuracy)

#%%
# Create a dictionary with the key "DEfrom scracth" and the list as its value
data_dict = {"epoch_sq":epoch_list,
             f"{target_data} train after Domain Adaptation": train_accuracy_values,
             f"{target_data} test after Domain Adaptation": test_accuracy_values

              }

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)

# Define the file path
file_path = target_data + "_traintest_acc_after_domain_adaptation_from_"+source_data+ ".xlsx"

# Write the DataFrame to an Excel file
df.to_excel(file_path, index=False)

print("Data has been stored in", file_path)

