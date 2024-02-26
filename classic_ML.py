#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:22:22 2024

@author: saiful
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:47:19 2023

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
given_data = "FR"
#%% UK
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_UK2.csv'
df_UK = pd.read_csv(file_path)
column_names = df_UK.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = df_UK['UK'].min()
max_value = df_UK['UK'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_UK = np.histogram(df_UK['UK'], bins=num_bins, range=(min_value, max_value))
print("Bin bin_edges_UK:", bin_edges_UK)

# Plot the histogram
plt.hist(df_UK['UK'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of UK column')
plt.xlabel('UK values')
plt.ylabel('Frequency')
plt.show()


#%% DE

file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv'
df_DE = pd.read_csv(file_path)
column_names = df_DE.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = df_DE['DE'].min()
max_value = df_DE['DE'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_DE = np.histogram(df_DE['DE'], bins=num_bins, range=(min_value, max_value))
print("Bin bin_edges_DE:", bin_edges_DE)

# Plot the histogram
plt.hist(df_DE['DE'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of DE column')
plt.xlabel('DE values')
plt.ylabel('Frequency')
plt.show()

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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# File paths
file_paths = [
    '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_UK2.csv',
    '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv',
    '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_FR2.csv',
]

# Read dataframes
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Concatenate all data to find the overall min and max
# all_data = pd.concat([df['UK'] for df in dfs] + [df['DE'] for df in dfs] + [df['FR'] for df in dfs])
all_data = pd.concat([df_UK['UK'], df_DE['DE'], df_FR['FR']])

# Calculate min and max values
min_value_of_all = all_data.min()   # 1e-05
max_value_of_all = df_UK['UK'].max()  # 0.98527  #  0.790560937


# =============================================================================
# #%% AFTER FIXING THE RANGE
# =============================================================================
#%% UK
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_UK2.csv'
df_UK = pd.read_csv(file_path)
column_names = df_UK.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = min_value_of_all
max_value = max_value_of_all

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_UK = np.histogram(df_UK['UK'], bins=num_bins, range=(min_value, max_value))
print("flag 1.21 Bin bin_edges_UK AFTER FIXING RANGE:", bin_edges_UK)

# Plot the histogram
plt.hist(df_UK['UK'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of UK column AFTER FIXING RANGE')
plt.xlabel('UK values')
plt.ylabel('Frequency')
plt.show()


#%% DE

file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv'
df_DE = pd.read_csv(file_path)
column_names = df_DE.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
# min_value = df_DE['DE'].min()
# max_value = df_DE['DE'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_DE = np.histogram(df_DE['DE'], bins=num_bins, range=(min_value, max_value))
print(" flag 1.21 Bin bin_edges_DE AFTER FIXING RANGE:", bin_edges_DE)

# Plot the histogram
plt.hist(df_DE['DE'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of DE column AFTER FIXING RANGE')
plt.xlabel('DE values')
plt.ylabel('Frequency')
plt.show()

#%% FR
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_FR2.csv'
df_FR = pd.read_csv(file_path)
column_names = df_FR.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
# min_value = df_FR['FR'].min()
# max_value = df_FR['FR'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_FR = np.histogram(df_FR['FR'], bins=num_bins, range=(min_value, max_value))
print(" flag 1.21 Bin bin_edges_FR AFTER FIXING RANGe:", bin_edges_FR)

# Plot the histogram
plt.hist(df_FR['FR'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of FR column AFTER FIXING RANGe')
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

selected_indices = [1, 3, 8, 9, 10, 13]
if given_data== 'FR':
    
    df = df_FR
    df.dropna(subset=['FR'], inplace=True)
    df = df.iloc[:, selected_indices + [-1]]
    X = df.drop('FR', axis=1)  # Features
    y = df['FR']  # Labels
elif given_data== 'DE':
    
    df = df_DE
    df.dropna(subset=['DE'], inplace=True)
    df = df.iloc[:, selected_indices + [-1]]
    X = df.drop('DE', axis=1)  # Features
    y = df['DE']  # Labels
    
elif given_data== 'UK':
    df = df_UK
    df.dropna(subset=['UK'], inplace=True)
    df = df.iloc[:, selected_indices + [-1]]
    X = df.drop('UK', axis=1)  # Features
    y = df['UK']  # Labels

# # Load the data
# df = df_FR
# # Split features and labels
# df.dropna(subset=['FR'], inplace=True)

#%%
# Assuming df is your DataFrame

# Selecting the index column and last column

print('flag 1.212: \ndf',df)
#%%
# X = df.drop('FR', axis=1)  # Features
# y = df['FR']  # Labels

# Define bins for class labels
# data_bin = [1.0000e-05, 1.6422e-01, 3.2843e-01, 4.9264e-01, 6.5685e-01, 8.2106e-01, 9.8527e-01]
data_bin = [1.00000000e-05, 1.31768490e-01, 2.63526979e-01, 3.95285468e-01, 5.27043958e-01, 6.58802447e-01, 7.90560937e-01]
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

#%%
# # Train the SVM model
# svm_model = SVC(kernel='linear')  # You can choose different kernels like 'rbf' or 'poly' as well
# svm_model.fit(X_train_scaled, y_train)

# # Make predictions
# y_pred = svm_model.predict(X_test_scaled)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

#%%
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier

# # Random Forest
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train_scaled, y_train)
# rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test_scaled))

# # Decision Tree
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train_scaled, y_train)
# dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test_scaled))

# # AdaBoost
# adaboost_model = AdaBoostClassifier(random_state=42)
# adaboost_model.fit(X_train_scaled, y_train)
# adaboost_accuracy = accuracy_score(y_test, adaboost_model.predict(X_test_scaled))

# # Gradient Boosting
# gb_model = GradientBoostingClassifier(random_state=42)
# gb_model.fit(X_train_scaled, y_train)
# gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test_scaled))

# # Print accuracies
# print("Random Forest Accuracy:", rf_accuracy)
# print("Decision Tree Accuracy:", dt_accuracy)
# print("AdaBoost Accuracy:", adaboost_accuracy)
# print("Gradient Boosting Accuracy:", gb_accuracy)
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

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

# Define the model
model = Wind_Model(input_size=X_train_tensor.shape[1], num_classes=len(y_train_tensor.unique())).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%%
train_accuracies = []
test_accuracies = []
epoch_list = []
epoch_list_train = []
# Train the model
num_epochs = 300
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
        
        # if epoch%5 == 0:
        #     # Print running loss and accuracy
        #     running_accuracy = correct_predictions / total_samples
        #     print(f"running Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {running_accuracy}")

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    train_accuracies.append(epoch_accuracy)
    epoch_list_train.append(epoch + 1)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Train Accuracy: {epoch_accuracy}")
    
        # Check if epoch is multiple of 5 for evaluating on test data
    if (epoch + 1) % 1 == 0:
        model.eval()
        predictions = []
        true_labels = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.unsqueeze(2))
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
    
        test_accuracy = accuracy_score(true_labels, predictions)
        test_accuracies.append(test_accuracy)
        epoch_list.append(epoch + 1)
        print("Test Accuracy:", test_accuracy)
        model.train()  # Switch back to training mode after evaluation

torch.save(model.state_dict(), given_data+'_wind_model.pth')

# Plotting
epochs = range(1, num_epochs + 1)
plt.plot(epochs, [acc * 100 for acc in train_accuracies], label='Training Accuracy')
# plt.plot(epochs[4::5], [acc * 100 for acc in test_accuracies], 'ro', label='Test Accuracy')  # Plotting test accuracy every 5 epochs
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.legend()
plt.ylim(0, 100)  # Set y-axis range from 0 to 100
plt.show()
#%%

# Create a dictionary with the key "DEfrom scracth" and the list as its value
data_dict = {"epoch_sq":epoch_list_train,
             f"{given_data} train from scratch": train_accuracies,
             f"{given_data} test from scratch": test_accuracies
              }

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)

# Define the file path
file_path = given_data + "_train_acc_from_scratch"+ ".xlsx"

# Write the DataFrame to an Excel file
df.to_excel(file_path, index=False)

print("Data has been stored in", file_path)
#%%
# Evaluate the model
model.eval()
predictions = []
true_labels = []
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs.unsqueeze(2)).to(device)
    _, predicted = torch.max(outputs, 1)
    predictions.extend(predicted.tolist())
    true_labels.extend(labels.tolist())

test_accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy on {given_data}:", test_accuracy)



#%% now load the pretrained DE model and test it with FR dataset